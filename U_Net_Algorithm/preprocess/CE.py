import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Disable OpenCV multi-threading to prevent Windows CPU deadlocks with TensorFlow
cv2.setNumThreads(1)


# ==========================================
# 1. DROPFILTER PLUS U-NET ARCHITECTURE
# ==========================================
class DoubleConvDropFilter(layers.Layer):
    def __init__(self, filters, drop_rate=0.2, **kwargs):
        super(DoubleConvDropFilter, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.drop = layers.SpatialDropout2D(drop_rate)

        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')

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

        self.up3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.dec3 = DoubleConvDropFilter(128, drop_rate=0.2)

        self.up2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.dec2 = DoubleConvDropFilter(64, drop_rate=0.2)

        self.up1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
        self.dec1 = DoubleConvDropFilter(32, drop_rate=0.1)

        self.final_conv = layers.Conv2D(out_channels, 1, activation='sigmoid')

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
# 2. DATASET PIPELINE
# ==========================================
class RetinopathyDataSequence(tf.keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=2, image_size=(256, 256), shuffle=True, **kwargs):
        super(RetinopathyDataSequence, self).__init__(**kwargs)

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.image_dir = os.path.join(root_dir, 'image')

        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if
                                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])
        else:
            self.image_files = []

        if len(self.image_files) == 0:
            raise ValueError(f"No valid images found in directory: {self.image_dir}")

        self.classes = ['EX', 'HE', 'MA', 'SE']
        self.indexes = np.arange(len(self.image_files))
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.floor(len(self.image_files) / self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if len(batch_indexes) == 0:
            batch_indexes = self.indexes[-self.batch_size:]

        batch_files = [self.image_files[k] for k in batch_indexes]

        images = []
        masks = []

        for img_name in batch_files:
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255.0
            images.append(image)

            class_masks = []
            for cls in self.classes:
                mask_path = os.path.join(self.root_dir, cls, img_name)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                        mask = (mask > 127).astype(np.float32)
                    else:
                        mask = np.zeros(self.image_size, dtype=np.float32)
                else:
                    mask = np.zeros(self.image_size, dtype=np.float32)
                class_masks.append(mask)

            combined_mask = np.stack(class_masks, axis=-1)
            masks.append(combined_mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


# ==========================================
# 3. COMBINED BCE + DICE LOSS & METRICS
# ==========================================
class SegmentationLossAndMetrics:
    def dice_coef(self, y_true, y_pred, smooth=1e-6):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def combined_bce_dice_loss(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = 1.0 - self.dice_coef(y_true, y_pred)
        return tf.reduce_mean(bce) + dice

    def pixel_accuracy(self, y_true, y_pred):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
        correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_bin), tf.float32))
        total = tf.cast(tf.size(y_true), tf.float32)
        return correct / total

    def f1_score(self, y_true, y_pred, smooth=1e-6):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred_bin)
        total_sum = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin)
        return (2.0 * intersection + smooth) / (total_sum + smooth)


# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================
class SegmentationTrainer:
    def __init__(self, spath, fpath, epochs=60, batch_size=2, lr=3e-4):
        self.spath = spath
        self.fpath = fpath
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.evaluator = SegmentationLossAndMetrics()
        self.model = DropFilterPlusUNet(out_channels=4)

        self.model.build((None, 256, 256, 3))
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=self.lr, weight_decay=1e-4),
            loss=self.evaluator.combined_bce_dice_loss,
            metrics=[self.evaluator.pixel_accuracy, self.evaluator.f1_score]
        )

        self.prepare_dataloaders()

    def prepare_dataloaders(self):
        self.train_gen = RetinopathyDataSequence(os.path.join(self.spath, 'train'), batch_size=self.batch_size,
                                                 image_size=(256, 256))
        self.valid_gen = RetinopathyDataSequence(os.path.join(self.spath, 'valid'), batch_size=self.batch_size,
                                                 image_size=(256, 256), shuffle=False)
        self.test_gen = RetinopathyDataSequence(os.path.join(self.spath, 'test'), batch_size=self.batch_size,
                                                image_size=(256, 256), shuffle=False)

    def execute_training(self):
        print("Training Started (Standard FP32 Mode)...")

        checkpoint = callbacks.ModelCheckpoint(
            filepath=self.fpath,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )

        early_stop = callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=10,
            start_from_epoch=5,
            verbose=0
        )

        self.model.fit(
            self.train_gen,
            validation_data=self.valid_gen,
            epochs=self.epochs,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )

    def run_testing(self):
        print("\n" + "=" * 45)
        print("EVALUATING MODEL ON TEST DATASET...")
        if os.path.exists(self.fpath):
            self.model.load_weights(self.fpath)

        results = self.model.evaluate(self.test_gen, verbose=0)

        print("=" * 45)
        print(f"FINAL TEST PIXEL ACCURACY: {results[1] * 100:.2f}%")
        print(f"FINAL TEST F1-SCORE      : {results[2] * 100:.2f}%")
        print("=" * 45)


# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
if __name__ == '__main__':
    spath = r"..\Output\contrast_enhancement"
    fpath = r"./dropfilter_plus_unet_best.weights.h5"

    trainer = SegmentationTrainer(
        spath=spath,
        fpath=fpath,
        epochs=60,
        batch_size=2,
        lr=3e-4
    )

    trainer.execute_training()
    trainer.run_testing()