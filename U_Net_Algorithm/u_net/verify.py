import os
import cv2
import numpy as np

spath = r"..\Output\data_png"
classes = ['EX', 'HE', 'MA', 'SE']

image_dir = os.path.join(spath, 'image')

print("=" * 50)
print("PATH CHECK & DEBUGGING")
print("=" * 50)

# 1. Image Folder Check
if os.path.exists(image_dir):
    all_imgs = os.listdir(image_dir)
    print(f"[✓] Image Directory Exists: {image_dir}")
    print(f"    Total files in 'image' folder: {len(all_imgs)}")
    if len(all_imgs) > 0:
        print(f"    Sample Image Filename: {all_imgs[0]}")
else:
    print(f"[X] Image Directory NOT Found: {image_dir}")

# 2. Classes Folders & Masks Check
for cls in classes:
    cls_dir = os.path.join(spath, cls)
    print("-" * 40)
    if os.path.exists(cls_dir):
        masks = os.listdir(cls_dir)
        print(f"[✓] Class Folder '{cls}' Found.")
        print(f"    Total masks in '{cls}' folder: {len(masks)}")

        if len(masks) > 0:
            sample_mask_path = os.path.join(cls_dir, masks[0])
            mask_img = cv2.imread(sample_mask_path, cv2.IMREAD_UNCHANGED)

            if mask_img is None:
                print(f"    [X] OpenCV failed to read mask file: {sample_mask_path}")
            else:
                print(f"    Mask Image Shape: {mask_img.shape}")
                print(f"    Mask Data Type  : {mask_img.dtype}")
                print(f"    Min Pixel Value : {np.min(mask_img)}")
                print(f"    Max Pixel Value : {np.max(mask_img)}")
                print(f"    Non-Zero Count  : {np.count_nonzero(mask_img)}")
    else:
        print(f"[X] Class Folder '{cls}' NOT Found: {cls_dir}")
print("=" * 50)