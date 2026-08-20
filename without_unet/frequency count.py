import os
import csv
import cv2
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.morphology import skeletonize
from scipy.spatial import distance_matrix, cKDTree
from concurrent.futures import ProcessPoolExecutor


class ImageLoader:
    """Handles multi-format image loading with Unicode path support."""

    def load_grayscale(self, file_path: Path) -> np.ndarray:
        try:
            img_array = np.fromfile(str(file_path), dtype=np.uint8)
            if img_array.size > 0:
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
        except Exception:
            pass

        with Image.open(file_path) as pil_img:
            return np.array(pil_img.convert("L"))


class SkeletonExtractor:
    """Extracts skeleton centerlines and key node positions."""

    def __init__(self, threshold: int = 127):
        self.threshold = threshold
        self.kernel = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)

    def extract_nodes(self, image: np.ndarray):
        _, binary = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary > 0)

        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, self.kernel)
        endpoints = (skeleton & (neighbor_count == 11))
        bifurcations = (skeleton & (neighbor_count >= 13))

        ep_y, ep_x = np.where(endpoints)
        bp_y, bp_x = np.where(bifurcations)

        all_x = np.concatenate([ep_x, bp_x])
        all_y = np.concatenate([ep_y, bp_y])
        nodes = np.column_stack((all_x, all_y))

        return skeleton, nodes, len(ep_x), len(bp_x)


class GabrielGraphBuilder:
    """Constructs a Gabriel Graph using O(N) memory KD-Tree queries."""

    def __init__(self, k: float = 0.1, x0: float = 50.0):
        self.k = k
        self.x0 = x0

    def _logistic_weight(self, dist: float) -> float:
        return 1.0 / (1.0 + np.exp(-self.k * (dist - self.x0)))

    def build(self, nodes: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        num_nodes = len(nodes)

        for i in range(num_nodes):
            G.add_node(i, pos=nodes[i])

        if num_nodes < 2:
            return G

        tree = cKDTree(nodes)
        dist_mat = distance_matrix(nodes, nodes)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                midpoint = (nodes[i] + nodes[j]) / 2.0
                radius = dist_mat[i, j] / 2.0

                neighbor_indices = tree.query_ball_point(midpoint, r=radius - 1e-7)
                in_circle = [idx for idx in neighbor_indices if idx != i and idx != j]

                if len(in_circle) == 0:
                    dist = dist_mat[i, j]
                    weight = self._logistic_weight(dist)
                    G.add_edge(i, j, distance=dist, weight=weight)

        return G


class AVFrequencyAnalyzer:
    """Calculates Arteriovenous structural frequency metrics from graph network."""

    def analyze(self, G: nx.Graph, skeleton: np.ndarray, num_ep: int, num_bp: int) -> dict:
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()

        # Calculate vessel pixel density
        vessel_pixels = np.count_nonzero(skeleton)
        total_pixels = skeleton.size
        vessel_density = (vessel_pixels / total_pixels) if total_pixels > 0 else 0.0

        # Node degree distributions (A/V branching frequency)
        degrees = [d for _, d in G.degree()]
        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = int(np.max(degrees)) if degrees else 0

        # Calculate edge lengths
        distances = [d.get("distance", 0.0) for _, _, d in G.edges(data=True)]
        total_vessel_length = float(np.sum(distances)) if distances else 0.0
        avg_segment_length = float(np.mean(distances)) if distances else 0.0

        return {
            "Total_Nodes": total_nodes,
            "Endpoints_Count": num_ep,
            "Bifurcations_Count": num_bp,
            "Total_Edges": total_edges,
            "Average_Node_Degree": round(avg_degree, 4),
            "Max_Branching_Degree": max_degree,
            "Vessel_Density": round(vessel_density, 6),
            "Total_Vessel_Length_px": round(total_vessel_length, 2),
            "Avg_Segment_Length_px": round(avg_segment_length, 2)
        }


class SingleFileAVProcessor:
    """Extracts A/V features for a single file task."""

    def __init__(self):
        self.loader = ImageLoader()
        self.extractor = SkeletonExtractor()
        self.builder = GabrielGraphBuilder()
        self.analyzer = AVFrequencyAnalyzer()

    def process_file(self, file_path: Path, input_dir: Path) -> dict:
        relative_path = file_path.relative_to(input_dir)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "Uncategorized"

        try:
            image = self.loader.load_grayscale(file_path)
            skeleton, nodes, num_ep, num_bp = self.extractor.extract_nodes(image)
            graph = self.builder.build(nodes)

            metrics = self.analyzer.analyze(graph, skeleton, num_ep, num_bp)
            metrics["File_Name"] = file_path.name
            metrics["Relative_Path"] = str(relative_path)
            metrics["Category"] = category

            print(f"Extracted A/V counts: {relative_path}")
            return metrics

        except Exception as err:
            print(f"Error {relative_path}: {err}")
            return None


def process_av_wrapper(task_args):
    file_path, input_dir = task_args
    processor = SingleFileAVProcessor()
    return processor.process_file(Path(file_path), Path(input_dir))


class AVFrequencyPipeline:
    """Orchestrates Arteriovenous Frequency Count extraction across datasets."""

    def __init__(self, input_dir: str, output_dir: str, valid_exts=(".png", ".jpg", ".jpeg", ".tif"),
                 max_workers: int = 4):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.valid_exts = tuple(ext.lower() for ext in valid_exts)
        self.max_workers = max_workers

    def run(self):
        if not self.input_dir.exists():
            print(f"Directory not found: {self.input_dir}")
            return

        tasks = []
        for root, _, files in os.walk(str(self.input_dir)):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.valid_exts:
                    tasks.append((str(file_path), str(self.input_dir)))

        print(f"Extracting A/V Frequency metrics for {len(tasks)} files...\n")

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            completed = executor.map(process_av_wrapper, tasks)
            for res in completed:
                if res:
                    results.append(res)

        if results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)

            # Reorder columns cleanly
            first_cols = ["Category", "File_Name", "Relative_Path"]
            other_cols = [c for c in df.columns if c not in first_cols]
            df = df[first_cols + other_cols]

            # Save CSV and Excel reports
            csv_path = self.output_dir / "av_frequency_counts.csv"
            excel_path = self.output_dir / "av_frequency_counts.xlsx"

            df.to_csv(csv_path, index=False)
            df.to_excel(excel_path, index=False)

            print(f"\nProcessing Complete!")
            print(f"CSV Report saved to: {csv_path}")
            print(f"Excel Report saved to: {excel_path}")


if __name__ == "__main__":
    INPUT_FOLDER = r"Output\vessel_centerline - Copy"
    OUTPUT_FOLDER = r"Output\av_frequency_analysis"

    pipeline = AVFrequencyPipeline(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        max_workers=4
    )
    pipeline.run()