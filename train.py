"""
Train the 14-joint pose regression model on the preprocessed COCO/MPII crops.

Model: MobileNetV1 (alpha=0.25, ImageNet weights) backbone taking 96x96
grayscale input, followed by a small regression head that outputs x, y in
[0, 1] for each joint in JOINT_ORDER (28 values).

Training runs in two stages: head only with the backbone frozen, then the
whole network at a lower learning rate. Unlabeled joints are excluded from
the loss using the mask saved by preprocess_dataset.py.

Usage: python train.py [--limit N] [--head-epochs N] [--full-epochs N]
"""
import argparse
import math
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers

DATA_PREFIX = "pose_dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_model.keras")

JOINT_ORDER = [
    "head", "neck",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
]
NUM_JOINTS = len(JOINT_ORDER)
# Index of each joint's mirror image, used when flipping a crop horizontally.
FLIP_PERM = [JOINT_ORDER.index(j.replace("l_", "#").replace("r_", "l_").replace("#", "r_"))
             for j in JOINT_ORDER]

IMG_SIZE = 96
BATCH_SIZE = 64
SHUFFLE_BUFFER = 20000

# Augmentation ranges
MAX_ROTATION_DEG = 25
SCALE_RANGE = (0.8, 1.2)
MAX_SHIFT_FRAC = 0.1
MAX_BRIGHTNESS_DELTA = 0.2
CONTRAST_RANGE = (0.7, 1.3)

HUBER_DELTA = 0.05  # in normalized coords (~5 px at 96x96)
PCK_THRESHOLD = 0.05  # a joint counts as correct within this distance (normalized)


def load_split(name, limit=None):
    """Return (images, labels, mask) for one split; images get a channel axis."""
    data = np.load(f"{DATA_PREFIX}_{name}.npz")
    images, labels, mask = data["images"], data["labels"], data["mask"]
    if limit:
        images, labels, mask = images[:limit], labels[:limit], mask[:limit]
    return images[..., None], labels, mask


def augment(image, label, mask):
    """Randomly flip, rotate, scale, shift, and relight a crop and its joints."""
    pts = tf.reshape(label, (NUM_JOINTS, 2))

    # Horizontal flip: mirror x and swap each left joint with its right joint.
    flip = tf.random.uniform(()) < 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    flipped = tf.gather(tf.stack([1 - pts[:, 0], pts[:, 1]], axis=1), FLIP_PERM)
    pts = tf.where(flip, flipped, pts)
    mask = tf.where(flip, tf.gather(mask, FLIP_PERM), mask)

    # Affine: points move by p' = s * R * (p - c) + c + t, around the crop center c.
    angle = tf.random.uniform((), -1, 1) * MAX_ROTATION_DEG * math.pi / 180
    scale = tf.random.uniform((), *SCALE_RANGE)
    shift = tf.random.uniform((2,), -MAX_SHIFT_FRAC, MAX_SHIFT_FRAC)
    cos, sin = tf.cos(angle), tf.sin(angle)
    rot = tf.reshape(tf.stack([cos, -sin, sin, cos]), (2, 2))
    pts = scale * tf.linalg.matvec(rot, pts - 0.5) + 0.5 + shift

    # The image transform maps each output pixel back to its input pixel:
    # p = A (p' - c - t) + c, where A = R^-1 / s.
    inv = tf.reshape(tf.stack([cos, sin, -sin, cos]), (2, 2)) / scale
    center = tf.constant([IMG_SIZE / 2, IMG_SIZE / 2])
    offset = center - tf.linalg.matvec(inv, center + shift * IMG_SIZE)
    transform = tf.stack([inv[0, 0], inv[0, 1], offset[0],
                          inv[1, 0], inv[1, 1], offset[1], 0.0, 0.0])
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None], transforms=transform[None],
        output_shape=[IMG_SIZE, IMG_SIZE], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="CONSTANT")[0]

    # Joints pushed outside the crop no longer count as labeled.
    inside = tf.reduce_all((pts >= 0) & (pts <= 1), axis=1)
    mask = mask * tf.cast(inside, tf.float32)

    image = tf.image.random_brightness(image, MAX_BRIGHTNESS_DELTA * 255)
    image = tf.image.random_contrast(image, *CONTRAST_RANGE)
    image = tf.clip_by_value(image, 0, 255)

    return image, tf.reshape(pts, (-1,)), mask


def make_dataset(images, labels, mask, training):
    """Build a batched tf.data pipeline; targets are labels and mask concatenated."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels, mask))
    ds = ds.map(lambda i, l, m: (tf.cast(i, tf.float32), l, m))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda i, l, m: (i, tf.concat([l, m], axis=0)))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_backbone():
    """MobileNetV1-0.25 for 1-channel input, initialized from ImageNet weights.

    The pretrained first conv expects RGB, so its kernel is summed over the
    color channels. That gives exactly the same output as feeding the
    grayscale image replicated to 3 channels, without the extra input buffer.
    """
    pretrained = keras.applications.MobileNet(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=0.25, include_top=False, weights="imagenet")
    backbone = keras.applications.MobileNet(
        input_shape=(IMG_SIZE, IMG_SIZE, 1), alpha=0.25, include_top=False, weights=None)

    first_conv = True
    for src, dst in zip(pretrained.layers, backbone.layers):
        weights = src.get_weights()
        if first_conv and isinstance(src, layers.Conv2D):
            weights[0] = weights[0].sum(axis=2, keepdims=True)
            first_conv = False
        dst.set_weights(weights)
    return backbone


def build_model():
    """Backbone plus regression head: raw 0-255 grayscale in, 28 coords in [0, 1] out.

    The head flattens a reduced 3x3 feature map instead of global average
    pooling, since pooling would throw away where in the frame features are.
    """
    backbone = build_backbone()
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # MobileNet expects [-1, 1]
    x = backbone(x)
    x = layers.Conv2D(64, 1, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2 * NUM_JOINTS, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="pose_model"), backbone


def split_target(y_true, y_pred):
    """Return (true points, predicted points, mask) shaped (N, joints, 2 / -)."""
    labels, mask = y_true[:, :2 * NUM_JOINTS], y_true[:, 2 * NUM_JOINTS:]
    return (tf.reshape(labels, (-1, NUM_JOINTS, 2)),
            tf.reshape(y_pred, (-1, NUM_JOINTS, 2)), mask)


def masked_huber_loss(y_true, y_pred):
    """Huber loss per coordinate, averaged over labeled joints only."""
    true_pts, pred_pts, mask = split_target(y_true, y_pred)
    err = tf.abs(true_pts - pred_pts)
    huber = tf.where(err < HUBER_DELTA, 0.5 * err ** 2, HUBER_DELTA * (err - 0.5 * HUBER_DELTA))
    per_joint = tf.reduce_sum(huber, axis=2)
    return tf.reduce_sum(per_joint * mask) / tf.maximum(tf.reduce_sum(mask), 1)


def joint_distances(y_true, y_pred):
    """Return (per-joint distance in normalized coords, mask)."""
    true_pts, pred_pts, mask = split_target(y_true, y_pred)
    return tf.norm(true_pts - pred_pts, axis=2), mask


def mean_px_error(y_true, y_pred):
    """Mean distance between predicted and true joints, in 96x96 crop pixels."""
    dist, mask = joint_distances(y_true, y_pred)
    return IMG_SIZE * tf.reduce_sum(dist * mask) / tf.maximum(tf.reduce_sum(mask), 1)


def pck(y_true, y_pred):
    """Fraction of labeled joints predicted within PCK_THRESHOLD of the truth."""
    dist, mask = joint_distances(y_true, y_pred)
    hits = tf.cast(dist < PCK_THRESHOLD, tf.float32) * mask
    return tf.reduce_sum(hits) / tf.maximum(tf.reduce_sum(mask), 1)


def train_stage(model, train_ds, val_ds, epochs, lr, name):
    """Compile and fit for one stage, keeping the weights with the best val loss."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=masked_huber_loss,
                  metrics=[mean_px_error, pck])
    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, f"history_{name}.csv")),
    ]
    print(f"--- Stage: {name} ({epochs} epochs, lr={lr}) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


def report(model, images, labels, mask):
    """Print per-joint val error and save a grid of predictions over crops."""
    pred = model.predict(images, batch_size=256, verbose=0)
    dist = np.linalg.norm((pred - labels).reshape(-1, NUM_JOINTS, 2), axis=2)

    print("Per-joint val error (px at 96x96) / PCK:")
    for j, name in enumerate(JOINT_ORDER):
        d = dist[mask[:, j] > 0, j]
        print(f"  {name:<11} {IMG_SIZE * d.mean():5.2f} px   {(d < PCK_THRESHOLD).mean():.3f}")

    # True joints in green, predictions in red, every crop scaled up 2x.
    tile, cols, rows = 2 * IMG_SIZE, 6, 4
    grid = Image.new("RGB", (cols * tile, rows * tile))
    for k, i in enumerate(np.random.RandomState(0).choice(len(images), cols * rows, replace=False)):
        crop = Image.fromarray(images[i, ..., 0]).convert("RGB").resize((tile, tile))
        draw = ImageDraw.Draw(crop)
        for j in range(NUM_JOINTS):
            px, py = pred[i, 2 * j] * tile, pred[i, 2 * j + 1] * tile
            if mask[i, j]:
                tx, ty = labels[i, 2 * j] * tile, labels[i, 2 * j + 1] * tile
                draw.line((tx, ty, px, py), fill=(255, 255, 0))
                draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=(0, 255, 0))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 0, 0))
        grid.paste(crop, ((k % cols) * tile, (k // cols) * tile))
    path = os.path.join(MODEL_DIR, "val_predictions.png")
    grid.save(path)
    print(f"Saved prediction grid to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only use the first N crops of each split (for testing)")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--full-epochs", type=int, default=40)
    args = parser.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    train = load_split("train", args.limit)
    val = load_split("val", args.limit)
    train_ds, val_ds = make_dataset(*train, training=True), make_dataset(*val, training=False)
    print(f"Train: {len(train[0])} crops, val: {len(val[0])} crops")

    model, backbone = build_model()
    model.summary()

    backbone.trainable = False
    train_stage(model, train_ds, val_ds, args.head_epochs, 1e-3, "head")
    backbone.trainable = True
    train_stage(model, train_ds, val_ds, args.full_epochs, 1e-4, "full")

    report(model, *val)


if __name__ == "__main__":
    main()
