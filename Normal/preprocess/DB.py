import os
import cv2
import random


class DataBalancer:
    def __init__(self, target_per_class=None):
        """
        :param target_per_class: Explicit image count per subfolder.
                                 If None, automatically matches the largest class count.
        """
        self.target_per_class = target_per_class

    def augment_image(self, image):
        """Applies random rotations (90, 180, 270 deg) and flips (horizontal, vertical)."""
        augmented = image.copy()

        # Random horizontal flip (50% chance)
        if random.choice([True, False]):
            augmented = cv2.flip(augmented, 1)

        # Random vertical flip (50% chance)
        if random.choice([True, False]):
            augmented = cv2.flip(augmented, 0)

        # Random rotation (0, 90, 180, or 270 degrees)
        rot_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
        if rot_code is not None:
            augmented = cv2.rotate(augmented, rot_code)

        return augmented

    def balance_folder(self, input_folder, output_folder):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

        # If no subfolders exist, handle the input_folder directly
        if not subfolders:
            subfolders = ['']

        # Count existing valid images per subfolder
        folder_files = {}
        for sub in subfolders:
            target_path = os.path.join(input_folder, sub) if sub else input_folder
            files = [f for f in os.listdir(target_path) if f.lower().endswith(valid_extensions)]
            if files:
                folder_files[sub] = files

        if not folder_files:
            print("No images found in the input folder.")
            return

        # Determine target count per class
        max_count = max(len(files) for files in folder_files.values())
        target_count = self.target_per_class if self.target_per_class else max_count

        print(f"Target images per class set to: {target_count}\n")

        for sub, files in folder_files.items():
            current_input_dir = os.path.join(input_folder, sub) if sub else input_folder
            current_output_dir = os.path.join(output_folder, sub) if sub else output_folder

            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

            current_count = len(files)

            # 1. Copy all original preprocessed images to the output folder
            for file_name in files:
                src_path = os.path.join(current_input_dir, file_name)
                dst_path = os.path.join(current_output_dir, file_name)
                img = cv2.imread(src_path)
                if img is not None:
                    cv2.imwrite(dst_path, img)

            # 2. Augment images if count is below target_count
            if current_count < target_count:
                needed = target_count - current_count
                print(f"[{sub if sub else 'Root'}] {current_count} images -> Adding {needed} augmented PNGs...")

                aug_idx = 0
                while aug_idx < needed:
                    random_file = random.choice(files)
                    img_path = os.path.join(current_input_dir, random_file)
                    img = cv2.imread(img_path)

                    if img is None:
                        continue

                    aug_img = self.augment_image(img)

                    # Save augmented file as PNG
                    base_name = os.path.splitext(random_file)[0]
                    aug_file_name = f"{base_name}_aug_{aug_idx + 1}.png"
                    aug_output_path = os.path.join(current_output_dir, aug_file_name)

                    cv2.imwrite(aug_output_path, aug_img)
                    aug_idx += 1
            else:
                print(f"[{sub if sub else 'Root'}] {current_count} images -> Already balanced.")


if __name__ == "__main__":
    INPUT_DIR = "..\\Output\\contrast_enhancement"
    OUTPUT_DIR = "..\\Output\\data_balancing"


    balancer = DataBalancer()
    balancer.balance_folder(INPUT_DIR, OUTPUT_DIR)

    print("\nData balancing complete! Files saved to:", OUTPUT_DIR)