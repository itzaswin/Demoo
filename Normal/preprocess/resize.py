import os
import cv2


class ImageResizer:
    def __init__(self, input_dir="..\\dataset", output_dir="..\\Output\\01_resize", target_size=(512, 512)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
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

                    resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_path, resized_img)


if __name__ == "__main__":
    resizer = ImageResizer(input_dir="..\\dataset", output_dir="..\\Output\\01_resize")
    resizer.run()