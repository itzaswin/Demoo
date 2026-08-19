import os
import cv2


class ContrastEnhancer:
    def __init__(self, input_dir="..\\Output\\02_denoised", output_dir="..\\Output\\03_contrast_enhanced", clip_limit=2.0, grid_size=(8, 8)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

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

                    if len(img.shape) == 3:
                        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
                        enhanced_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    else:
                        enhanced_img = self.clahe.apply(img)

                    cv2.imwrite(output_path, enhanced_img)


if __name__ == "__main__":
    enhancer = ContrastEnhancer(input_dir="..\\Output\\02_denoised", output_dir="..\\Output\\03_contrast_enhanced")
    enhancer.run()