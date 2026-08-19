import os
import cv2
import numpy as np


class NoiseRemover:
    def __init__(self, input_dir="..\\Output\\01_resize", output_dir="..\\Output\\02_denoised", niter=10, kappa=30, gamma=0.1):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.niter = niter
        self.kappa = kappa
        self.gamma = gamma
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def _anisotropic_diffusion(self, image):
        img = image.astype('float32')
        for _ in range(self.niter):
            delta_N = np.roll(img, -1, axis=0) - img
            delta_S = np.roll(img, 1, axis=0) - img
            delta_E = np.roll(img, -1, axis=1) - img
            delta_W = np.roll(img, 1, axis=1) - img

            cN = np.exp(-(delta_N / self.kappa) ** 2)
            cS = np.exp(-(delta_S / self.kappa) ** 2)
            cE = np.exp(-(delta_E / self.kappa) ** 2)
            cW = np.exp(-(delta_W / self.kappa) ** 2)

            img += self.gamma * (cN * delta_N + cS * delta_S + cE * delta_E + cW * delta_W)

        return np.clip(img, 0, 255).astype('uint8')

    def run(self):
        for root, _, files in os.walk(self.input_dir):
            for file_name in files:
                if file_name.lower().endswith(self.valid_extensions):
                    input_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(root, self.input_dir)
                    current_output_dir = os.path.join(self.output_dir, relative_path)

                    if not os.path.exists(current_output_dir):
                        os.makedirs(current_output_dir)

                    base_name = os.path.splitext(file_name)[0]
                    output_path = os.path.join(current_output_dir, f"{base_name}.png")

                    img = cv2.imread(input_path)
                    if img is None:
                        continue

                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                    denoised_img = self._anisotropic_diffusion(gray_img)
                    cv2.imwrite(output_path, denoised_img)


if __name__ == "__main__":
    denoiser = NoiseRemover(input_dir="..\\Output\\01_resize", output_dir="..\\Output\\02_denoised")
    denoiser.run()