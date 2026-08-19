import os
import cv2
import numpy as np


class VesselCenterlineExtractor:
    def zhang_suen_thinning(self, image):
        img = image.copy()
        img[img > 0] = 1

        while True:
            marker = np.zeros(img.shape, dtype=np.uint8)

            p2 = np.roll(img, -1, axis=0)
            p3 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
            p4 = np.roll(img, -1, axis=1)
            p5 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
            p6 = np.roll(img, 1, axis=0)
            p7 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
            p8 = np.roll(img, 1, axis=1)
            p9 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)

            # Sub-iteration 1
            A = ((p2 == 0) & (p3 == 1)).astype(int) + \
                ((p3 == 0) & (p4 == 1)).astype(int) + \
                ((p4 == 0) & (p5 == 1)).astype(int) + \
                ((p5 == 0) & (p6 == 1)).astype(int) + \
                ((p6 == 0) & (p7 == 1)).astype(int) + \
                ((p7 == 0) & (p8 == 1)).astype(int) + \
                ((p8 == 0) & (p9 == 1)).astype(int) + \
                ((p9 == 0) & (p2 == 1)).astype(int)

            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

            cond1 = (img == 1)
            cond2 = (B >= 2) & (B <= 6)
            cond3 = (A == 1)
            cond4 = (p2 * p4 * p6 == 0)
            cond5 = (p4 * p6 * p8 == 0)

            marker[cond1 & cond2 & cond3 & cond4 & cond5] = 1
            img[marker == 1] = 0

            # Sub-iteration 2
            marker.fill(0)
            p2 = np.roll(img, -1, axis=0)
            p3 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
            p4 = np.roll(img, -1, axis=1)
            p5 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
            p6 = np.roll(img, 1, axis=0)
            p7 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
            p8 = np.roll(img, 1, axis=1)
            p9 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)

            A = ((p2 == 0) & (p3 == 1)).astype(int) + \
                ((p3 == 0) & (p4 == 1)).astype(int) + \
                ((p4 == 0) & (p5 == 1)).astype(int) + \
                ((p5 == 0) & (p6 == 1)).astype(int) + \
                ((p6 == 0) & (p7 == 1)).astype(int) + \
                ((p7 == 0) & (p8 == 1)).astype(int) + \
                ((p8 == 0) & (p9 == 1)).astype(int) + \
                ((p9 == 0) & (p2 == 1)).astype(int)

            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

            cond4_2 = (p2 * p4 * p8 == 0)
            cond5_2 = (p2 * p6 * p8 == 0)

            marker[cond1 & cond2 & cond3 & cond4_2 & cond5_2] = 1
            img[marker == 1] = 0

            if not np.any(marker):
                break

        return (img * 255).astype(np.uint8)

    def extract_centerline(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Handle both input types:
        # If input has a white background (mean > 127), invert it so vessels become white (255)
        if np.mean(gray) > 127:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Apply Zhang-Suen Iterative Thinning
        thin_vessels = self.zhang_suen_thinning(binary)

        # Output format: Black background (0) with White centerlines (255)
        output = np.zeros_like(thin_vessels, dtype=np.uint8)
        output[thin_vessels > 0] = 255

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

                    centerline_img = self.extract_centerline(img)
                    cv2.imwrite(output_path, centerline_img)


if __name__ == "__main__":
    INPUT_DIR = "..\\Output\\vessel_segmentation"
    OUTPUT_DIR = "..\\Output\\vessel_centerline"

    extractor = VesselCenterlineExtractor()
    extractor.process_folder(INPUT_DIR, OUTPUT_DIR)