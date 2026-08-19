import os
import cv2
import numpy as np


class LesionBoundaryPipeline:

  def __init__(self, output_size=(512, 512)):
    self.output_size = output_size
    self.classes = ["EX", "HE", "MA", "SE"]

    # Boundary colors (BGR for OpenCV)
    self.class_colors = {
        "EX": (0, 255, 255),  # Yellow
        "HE": (0, 0, 255),  # Red
        "MA": (255, 0, 0),  # Blue
        "SE": (0, 255, 0),  # Green
    }

  def extract_active_linear_contours(
      self, binary_mask, connectivity=8, min_area=5
  ):
    """Diagonal Active Linear Contour (D_Conqour) algorithm."""
    if np.sum(binary_mask) == 0:
      return []

    # Clean noise and small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find external contours representing active linear boundaries
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    valid_contours = []
    for cnt in contours:
      if cv2.contourArea(cnt) >= min_area:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        valid_contours.append(approx_cnt)

    return valid_contours

  def process_boundaries(self, seg_dir, output_dir):
    """Uses ONLY lession_segmentation folder as input."""
    if not os.path.exists(seg_dir):
      raise FileNotFoundError(f"[X] Segmentation folder not found: {seg_dir}")

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

    # Get DR category subfolders (e.g., '1. No DR signs', etc.)
    subfolders = [
        f
        for f in os.listdir(seg_dir)
        if os.path.isdir(os.path.join(seg_dir, f))
    ]

    for category_folder in subfolders:
      category_path = os.path.join(seg_dir, category_folder)
      overlay_dir = os.path.join(category_path, "overlay")
      masks_dir = os.path.join(category_path, "masks")

      if not os.path.exists(overlay_dir):
        continue

      save_boundary_dir = os.path.join(output_dir, category_folder)
      os.makedirs(save_boundary_dir, exist_ok=True)

      image_files = [
          f
          for f in os.listdir(overlay_dir)
          if f.lower().endswith(valid_exts)
      ]
      print(
          f"\n[Processing Boundaries] Folder: {category_folder} ({len(image_files)} images)"
      )

      for img_name in image_files:
        # Load image directly from lession_segmentation/overlay
        overlay_img_path = os.path.join(overlay_dir, img_name)
        base_img = cv2.imread(overlay_img_path)

        if base_img is None:
          continue

        h, w = base_img.shape[:2]
        boundary_canvas = base_img.copy()

        # Process active linear boundaries for each lesion class
        for cls in self.classes:
          mask_path = os.path.join(masks_dir, cls, img_name)

          if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is not None and np.any(mask > 0):
              if mask.shape != (h, w):
                mask = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST
                )

              # Compute active linear contours (D_Conqour)
              contours = self.extract_active_linear_contours(mask)

              # Draw active boundary lines on the overlay canvas
              color = self.class_colors[cls]
              cv2.drawContours(
                  boundary_canvas,
                  contours,
                  -1,
                  color,
                  thickness=2,
                  lineType=cv2.LINE_AA,
              )

        # Save boundary result
        save_path = os.path.join(save_boundary_dir, img_name)
        cv2.imwrite(save_path, boundary_canvas)

    print(f"\n[✓] Lesion Boundaries saved to: {output_dir}")


# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
  # ONLY segmentation directory needed as input
  segmentation_dir = ("..\\Output\\lession_segmentation")
  boundary_output_dir = ("..\\Output\\lession_boundary")

  boundary_extractor = LesionBoundaryPipeline()
  boundary_extractor.process_boundaries(
      seg_dir=segmentation_dir, output_dir=boundary_output_dir
  )