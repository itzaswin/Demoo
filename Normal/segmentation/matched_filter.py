import os
import cv2
import numpy as np


class VesselSegmenter:
    def __init__(self, thickness=1):
        self.thickness = thickness

    def create_fov_mask(self, gray_image):
        _, mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.erode(mask, kernel, iterations=2)
        return mask

    def segment_vessels(self, image):
        # Step 1: Use Green Channel (highest vessel contrast in fundus images)
        if len(image.shape) == 3:
            green_channel = image[:, :, 1]
        else:
            green_channel = image.copy()

        # Step 2: Create FOV Mask to remove outer border artifacts
        fov_mask = self.create_fov_mask(green_channel)

        # Step 3: Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)

        # Step 4: Morphological Top-Hat Transformation to isolate linear vessel structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        top_hat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

        # Step 5: Clean noise using Fast N-Means Denoising
        denoised = cv2.fastNlMeansDenoising(top_hat, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Step 6: Thresholding to produce binary vessels
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply FOV Mask to force anything outside the fundus circle to pure background
        binary = cv2.bitwise_and(binary, binary, mask=fov_mask)

        # Step 7: Skeletonize/Thin to maintain exact vessel thickness (1 or 2 pixels)
        skel = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_binary = binary.copy()

        while True:
            eroded = cv2.erode(temp_binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_binary, temp)
            skel = cv2.bitwise_or(skel, temp)
            temp_binary = eroded.copy()
            if cv2.countNonZero(temp_binary) == 0:
                break

        if self.thickness > 1:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.thickness, self.thickness))
            vessels = cv2.dilate(skel, dilate_kernel)
        else:
            vessels = skel

        # Step 8: Output Format -> Pure White Background (255) with Black Vessels (0)
        output = np.full_like(vessels, 255, dtype=np.uint8)
        output[vessels > 0] = 0

        return output

    def process_folder(self, input_folder, output_folder):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

        for root, _, files in os.walk(input_folder):
            for file_name in files:
                if file_name.lower().endswith(valid_extensions):
                    input_path = os.path.join(root, file_name)

                    relative_path = os.path.relpath(root, input_folder)
                    current_output_dir = os.path.join(output_folder, relative_path)

                    if not os.path.exists(current_output_dir):
                        os.makedirs(current_output_dir)

                    base_name = os.path.splitext(file_name)[0]
                    png_file_name = f"{base_name}.png"
                    output_path = os.path.join(current_output_dir, png_file_name)

                    img = cv2.imread(input_path)
                    if img is None:
                        continue

                    segmented_img = self.segment_vessels(img)
                    cv2.imwrite(output_path, segmented_img)


if __name__ == "__main__":
    INPUT_DIR = "..\\Output\\data_balancing"
    OUTPUT_DIR = "..\\Output\\vessel_segmentation"

    segmenter = VesselSegmenter(thickness=1)
    segmenter.process_folder(INPUT_DIR, OUTPUT_DIR)