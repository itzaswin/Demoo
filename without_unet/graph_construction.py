import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


class GabrielGraphBuilder:
    """Extracts skeleton nodes and builds a graph represented as plain Python dictionaries."""

    def __init__(self, threshold=127, k=0.1, x0=50.0):
        import numpy as np
        self.threshold = threshold
        self.k = k
        self.x0 = x0
        self._neighbor_kernel = np.array([[1, 1, 1],
                                          [1, 10, 1],
                                          [1, 1, 1]], dtype=np.uint8)

    def _logistic_weight(self, dist):
        import numpy as np
        return 1.0 / (1.0 + np.exp(-self.k * (dist - self.x0)))

    def extract_nodes(self, image):
        import cv2
        import numpy as np
        from skimage.morphology import skeletonize

        _, binary = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary > 0)

        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, self._neighbor_kernel)

        endpoints = skeleton & (neighbor_count == 11)
        bifurcations = skeleton & (neighbor_count >= 13)

        ep_y, ep_x = np.where(endpoints)
        bp_y, bp_x = np.where(bifurcations)

        nodes = np.column_stack((np.concatenate([ep_x, bp_x]), np.concatenate([ep_y, bp_y])))
        return skeleton, nodes

    def build_graph(self, nodes):
        from scipy.spatial import cKDTree, distance_matrix

        graph = {
            "nodes": {i: nodes[i] for i in range(len(nodes))},
            "edges": []
        }

        num_nodes = len(nodes)
        if num_nodes < 2:
            return graph

        tree = cKDTree(nodes)
        dist_mat = distance_matrix(nodes, nodes)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                midpoint = (nodes[i] + nodes[j]) / 2.0
                radius = dist_mat[i, j] / 2.0

                points_inside = [
                    idx for idx in tree.query_ball_point(midpoint, r=radius - 1e-7)
                    if idx != i and idx != j
                ]

                if not points_inside:
                    dist = float(dist_mat[i, j])
                    weight = float(self._logistic_weight(dist))
                    graph["edges"].append((i, j, {"distance": dist, "weight": weight}))

        return graph


class GraphVisualizer:
    """Handles image loading and rendering the graph overlaid on the original image."""

    def __init__(self, edge_color=(255, 255, 0), node_color=(0, 0, 255), node_radius=2, line_thickness=1):
        self.edge_color = edge_color
        self.node_color = node_color
        self.node_radius = node_radius
        self.line_thickness = line_thickness

    def load_grayscale(self, file_path):
        import cv2
        import numpy as np
        from PIL import Image

        file_path = Path(file_path).resolve()

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

    def render_and_save(self, graph, base_image, output_path):
        """Renders graph edges and nodes overlaid directly on the input image."""
        import cv2

        # Convert grayscale base image to 3-channel BGR image for colored drawings
        if len(base_image.shape) == 2:
            canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = base_image.copy()

        nodes_dict = graph["nodes"]
        edges_list = graph["edges"]

        # Draw graph edges on top of the base image
        for u, v, _ in edges_list:
            pt1 = tuple(nodes_dict[u].astype(int))
            pt2 = tuple(nodes_dict[v].astype(int))
            cv2.line(canvas, pt1, pt2, self.edge_color, self.line_thickness, cv2.LINE_AA)

        # Draw graph nodes on top
        for node_id, pos in nodes_dict.items():
            pt = tuple(pos.astype(int))
            cv2.circle(canvas, pt, self.node_radius, self.node_color, -1, cv2.LINE_AA)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        success, buffer = cv2.imencode(".png", canvas)
        if success:
            buffer.tofile(str(output_path))


class VesselGraphPipeline:
    """Orchestrates dataset scanning and parallel worker distribution."""

    def __init__(
            self,
            input_dir,
            output_dir,
            valid_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
            max_workers=2
    ):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.valid_extensions = tuple(ext.lower() for ext in valid_extensions)
        self.max_workers = max_workers

    def run(self):
        print(f"Current Working Dir : {Path.cwd()}")
        print(f"Target Input Dir    : {self.input_dir}")

        if not self.input_dir.exists():
            print(f"\n[ERROR] Directory does not exist: {self.input_dir}")
            return

        tasks = []
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.valid_extensions:
                tasks.append((str(file_path), str(self.input_dir), str(self.output_dir)))

        if not tasks:
            print(f"\n[WARNING] No matching images found in directory.")
            return

        print(f"Found {len(tasks)} files. Processing using {self.max_workers} processes...\n")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(_worker_task, tasks)


def _worker_task(task_args):
    file_path_str, input_dir_str, output_dir_str = task_args
    file_path = Path(file_path_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    relative_path = file_path.relative_to(input_dir)
    dest_path = (output_dir / relative_path).with_suffix(".png")

    builder = GabrielGraphBuilder()
    visualizer = GraphVisualizer()

    try:
        image = visualizer.load_grayscale(file_path)
        _, nodes = builder.extract_nodes(image)
        graph = builder.build_graph(nodes)

        # Pass the original loaded image as the background canvas
        visualizer.render_and_save(graph, image, dest_path)
        print(f"Processed image: {relative_path}")

    except Exception as err:
        print(f"Error processing {relative_path}: {err}")


if __name__ == "__main__":
    pipeline = VesselGraphPipeline(
        input_dir="Output/vessel_centerline",
        output_dir="Output/graph_construction",
        max_workers=2
    )
    pipeline.run()