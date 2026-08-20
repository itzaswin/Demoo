import os
from pathlib import Path
import cv2
import numpy as np


class DAL_DC_MultiColorBoundaryPipeline:

    def __init__(self, output_size=(512, 512)):
        self.output_size = output_size

        # self.class_colors = {
        #     "HE": {"draw_color": (0, 0, 255)},  # Red BGR
        #     "SE": {"draw_color": (0, 255, 0)},  # Green BGR
        #     "MA": {"draw_color": (255, 0, 0)},  # Blue BGR
        #     "EX": {"draw_color": (0, 255, 255)}  # Yellow BGR
        # }
        self.class_colors = {
            "HE": (0, 0, 255),  # Red
            "SE": (0, 255, 0),  # Green
            "MA": (255, 0, 0),  # Blue
            "EX": (0, 255, 255)  # Yellow
        }

    def dal_dc_extract_contours(self, binary_mask, min_area=1):

        if np.sum(binary_mask) == 0:
            return []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_indices = np.where(areas >= min_area)[0] + 1

        if len(valid_indices) == 0:
            return []

        clean_mask = np.isin(labels, valid_indices).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return contours

    def process_boundaries(self, seg_dir, output_dir, boundary_thickness=2, high_contrast=True, draw_labels=False):

        seg_path = Path(seg_dir).resolve()
        out_path = Path(output_dir).resolve()

        if not seg_path.exists():
            print(f" CRITICAL ERROR: Input folder does not exist:\n    {seg_path}")
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        category_folders = [f for f in seg_path.iterdir() if f.is_dir()]

        for category_folder in category_folders:
            save_boundary_dir = out_path / category_folder.name
            save_boundary_dir.mkdir(parents=True, exist_ok=True)

            all_files = list(category_folder.rglob("*"))
            image_files = [
                f for f in all_files
                if f.is_file() and f.suffix.lower() in valid_exts
            ]

            print(
                f"\n[DAL_DC Bounding Box Processing] Category: '{category_folder.name}' ({len(image_files)} files found)")

            for img_path in image_files:
                img = cv2.imread(str(img_path))

                if img is None:
                    continue

                boundary_canvas = img.copy()
                boxes_drawn = 0

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, global_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

                if np.any(global_mask > 0):
                    contours = self.dal_dc_extract_contours(global_mask, min_area=1)

                    if contours:
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            box_color = (0, 0, 255)

                            if high_contrast:
                                cv2.rectangle(
                                    boundary_canvas,
                                    (x - 1, y - 1),
                                    (x + w + 1, y + h + 1),
                                    (0, 0, 0),
                                    thickness=boundary_thickness + 2
                                )

                            # Main bounding box
                            cv2.rectangle(
                                boundary_canvas,
                                (x, y),
                                (x + w, y + h),
                                box_color,
                                thickness=boundary_thickness
                            )

                            boxes_drawn += 1

                if boxes_drawn > 0:
                    print(f"  Bounding boxes applied ({boxes_drawn} boxes) -> {img_path.name}")
                else:
                    print(f"  Base image without colored mask annotations -> {img_path.name}")

                save_file = save_boundary_dir / img_path.name
                cv2.imwrite(str(save_file), boundary_canvas)

        print(f"\n  Processing complete! Results saved at:\n    {out_path}")


if __name__ == "__main__":
    segmentation_dir = r"..\Output\lesion_segmentation"
    boundary_output_dir = r"..\Output\lesion_boundary"

    pipeline = DAL_DC_MultiColorBoundaryPipeline()
    pipeline.process_boundaries(
        seg_dir=segmentation_dir,
        output_dir=boundary_output_dir,
        boundary_thickness=1,
        high_contrast=True,
        draw_labels=False
    )