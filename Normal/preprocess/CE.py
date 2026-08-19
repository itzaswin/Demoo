import os
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, target_size=(512, 512), clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def resize(self, image):
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def anisotropic_diffusion(self, image, niter=10, kappa=50, gamma=0.1):
        img = image.astype('float32')
        for _ in range(niter):
            delta_N = np.roll(img, -1, axis=0) - img
            delta_S = np.roll(img, 1, axis=0) - img
            delta_E = np.roll(img, -1, axis=1) - img
            delta_W = np.roll(img, 1, axis=1) - img

            cN = np.exp(-(delta_N / kappa) ** 2)
            cS = np.exp(-(delta_S / kappa) ** 2)
            cE = np.exp(-(delta_E / kappa) ** 2)
            cW = np.exp(-(delta_W / kappa) ** 2)

            img += gamma * (cN * delta_N + cS * delta_S + cE * delta_E + cW * delta_W)

        return np.clip(img, 0, 255).astype('uint8')

    def apply_clahe(self, image):
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return self.clahe.apply(image)

    def process_folder(self, input_folder, output_folder):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        for root, _, files in os.walk(input_folder):
            for file_name in files:
                if file_name.lower().endswith(valid_extensions):
                    input_path = os.path.join(root, file_name)

                    relative_path = os.path.relpath(root, input_folder)
                    current_output_dir = os.path.join(output_folder, relative_path)

                    if not os.path.exists(current_output_dir):
                        os.makedirs(current_output_dir)

                    # Replace original file extension with .png
                    base_name = os.path.splitext(file_name)[0]
                    png_file_name = f"{base_name}.png"
                    output_path = os.path.join(current_output_dir, png_file_name)

                    img = cv2.imread(input_path)
                    if img is None:
                        continue

                    resized_img = self.resize(img)
                    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    denoised_img = self.anisotropic_diffusion(gray_img, niter=10, kappa=30)
                    enhanced_img = self.apply_clahe(denoised_img)

                    cv2.imwrite(output_path, enhanced_img)


if __name__ == "__main__":
    INPUT_DIR = "..\\dataset"
    OUTPUT_DIR = "..\\Output\\contrast_enhancement"

    processor = ImageProcessor(target_size=(512, 512))
    processor.process_folder(INPUT_DIR, OUTPUT_DIR)