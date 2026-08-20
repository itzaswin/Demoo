import os

# Disable TensorFlow oneDNN/MKL optimizations to prevent Conv2DTranspose crashes
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

cv2.setNumThreads(1)


# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
class DoubleConvDropFilter(layers.Layer):

  def __init__(self, filters, drop_rate=0.2, **kwargs):
    super(DoubleConvDropFilter, self).__init__(**kwargs)
    self.conv1 = layers.Conv2D(filters, 3, padding="same", use_bias=False)
    self.bn1 = layers.BatchNormalization()
    self.act1 = layers.Activation("relu")
    self.drop = layers.SpatialDropout2D(drop_rate)

    self.conv2 = layers.Conv2D(filters, 3, padding="same", use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.act2 = layers.Activation("relu")

  def call(self, inputs, training=None):
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.act1(x)
    x = self.drop(x, training=training)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.act2(x)
    return x


class DropFilterPlusUNet(models.Model):

  def __init__(self, out_channels=4, **kwargs):
    super(DropFilterPlusUNet, self).__init__(**kwargs)
    self.enc1 = DoubleConvDropFilter(32, drop_rate=0.1)
    self.pool1 = layers.MaxPooling2D((2, 2))

    self.enc2 = DoubleConvDropFilter(64, drop_rate=0.2)
    self.pool2 = layers.MaxPooling2D((2, 2))

    self.enc3 = DoubleConvDropFilter(128, drop_rate=0.2)
    self.pool3 = layers.MaxPooling2D((2, 2))

    self.bottleneck = DoubleConvDropFilter(256, drop_rate=0.3)

    self.up3 = layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding="same"
    )
    self.dec3 = DoubleConvDropFilter(128, drop_rate=0.2)

    self.up2 = layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding="same"
    )
    self.dec2 = DoubleConvDropFilter(64, drop_rate=0.2)

    self.up1 = layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding="same"
    )
    self.dec1 = DoubleConvDropFilter(32, drop_rate=0.1)

    self.final_conv = layers.Conv2D(
        out_channels, 1, activation="sigmoid", dtype="float32"
    )

  def call(self, inputs, training=None):
    e1 = self.enc1(inputs, training=training)
    p1 = self.pool1(e1)

    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)

    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)

    b = self.bottleneck(p3, training=training)

    d3 = self.up3(b)
    d3 = layers.concatenate([d3, e3], axis=-1)
    d3 = self.dec3(d3, training=training)

    d2 = self.up2(d3)
    d2 = layers.concatenate([d2, e2], axis=-1)
    d2 = self.dec2(d2, training=training)

    d1 = self.up1(d2)
    d1 = layers.concatenate([d1, e1], axis=-1)
    d1 = self.dec1(d1, training=training)

    return self.final_conv(d1)


# ==========================================
# 2. PREDICTION PIPELINE
# ==========================================
class LesionPredictionPipeline:

  def __init__(self, weights_path, image_size=(512, 512), threshold=0.5):
    self.image_size = image_size
    self.threshold = threshold
    self.classes = ["EX", "HE", "MA", "SE"]

    # BGR Color Mapping for classes
    self.class_colors = {
        "EX": (0, 255, 255),  # Yellow
        "HE": (0, 0, 255),  # Red
        "MA": (255, 0, 0),  # Blue
        "SE": (0, 255, 0),  # Green
    }

    # Re-build architecture explicitly before loading weights
    self.model = DropFilterPlusUNet(out_channels=len(self.classes))

    # Initialize sub-layers via dummy forward pass
    dummy_input = tf.zeros((1, *image_size, 3))
    _ = self.model(dummy_input, training=False)

    # Load weights
    if os.path.exists(weights_path):
      self.model.load_weights(weights_path)
      print(f"[✓] Weights successfully loaded from: {weights_path}")
    else:
      raise FileNotFoundError(f"[X] Weights file not found: {weights_path}")

  def run_prediction(self, image_dir, output_dir):
    if not os.path.exists(image_dir):
      raise FileNotFoundError(f"[X] Image directory not found: {image_dir}")

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

    for root, _, files in os.walk(image_dir):
      image_files = [f for f in files if f.lower().endswith(valid_exts)]
      if not image_files:
        continue

      # Recreate target folder structure
      rel_path = os.path.relpath(root, image_dir)
      current_output_dir = os.path.join(output_dir, rel_path)
      os.makedirs(current_output_dir, exist_ok=True)

      print(f"\nProcessing Folder: {rel_path} ({len(image_files)} images)")

      for img_name in image_files:
        img_path = os.path.join(root, img_name)
        original_img = cv2.imread(img_path)

        if original_img is None:
          print(f" -> Skipped unreadable file: {img_name}")
          continue

        orig_h, orig_w = original_img.shape[:2]

        # Preprocess input image
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(
            img_rgb, self.image_size, interpolation=cv2.INTER_AREA
        )
        norm_img = resized_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(norm_img, axis=0)

        # Predict probability map
        predictions = self.model(input_tensor, training=False).numpy()[0]

        # Pure black canvas for output mask (Background completely removed)
        pure_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # Plot predicted classes with their assigned BGR colors
        for i, cls in enumerate(self.classes):
          class_prob = predictions[:, :, i]
          binary_mask = (class_prob > self.threshold).astype(np.uint8) * 255
          resized_mask = cv2.resize(
              binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
          )

          mask_indices = resized_mask > 0
          if np.any(mask_indices):
            pure_mask[mask_indices] = self.class_colors[cls]

        # Save 4-color mask image with black background directly to output directory
        save_path = os.path.join(current_output_dir, img_name)
        cv2.imwrite(save_path, pure_mask)

    print(
        f"\n[✓] Lesion Segmentation predictions completed! Saved to:"
        f" {output_dir}"
    )


# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
  input_weights_path = "unet_best.weights.h5"
  image_dir = r"..\Output\data_balancing"
  output_dir = r"..\Output\lesion_segmentation"

  predictor = LesionPredictionPipeline(
      weights_path=input_weights_path, image_size=(512, 512), threshold=0.5
  )

  predictor.run_prediction(image_dir=image_dir, output_dir=output_dir)