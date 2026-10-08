"""
Train the 14-joint pose regression model on the preprocessed COCO/MPII crops.

Model: MobileNetV1 (alpha=0.25, ImageNet weights) backbone taking 96x96
grayscale input, followed by a small head that outputs x, y in [0, 1] for
each joint in JOINT_ORDER (28 values), then a visibility score in [0, 1] for
each joint (14 values): whether it is labeled and inside the crop.

Training runs in two stages: head only with the backbone frozen, then the
whole network at a lower learning rate. Unlabeled joints are excluded from
the coordinate loss using the mask saved by preprocess_dataset.py, and the
mask is the target for the visibility scores. Some crops are zoomed in on a
single joint so the model also sees partial bodies (see PARTIAL_PROB).

Training can be stopped at any point (Ctrl+C, closing the terminal) and
picks up where it left off when the same command is run again, losing at
most BACKUP_EVERY_STEPS batches of progress. Delete the run's folder to
start over.

Each run lives in its own folder under MODELS_ROOT (models/<run>), with a
copy of this script, so earlier models are kept for comparison.

Usage: python train.py --run NAME [--init-from RUN] [--limit N]
                       [--head-epochs N] [--full-epochs N]
"""
import argparse
import csv
import glob
import math
import os
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers

DATA_PREFIX = "pose_dataset"
MODELS_ROOT = "models"
DEFAULT_RUN = "v1"  # used by evaluate.py and live_demo.py when no run is given
MODEL_FILENAME = "pose_model.keras"
BACKUP_EVERY_STEPS = 200  # ~1 min of training on CPU

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
# Partial views: zoom in on one labeled joint so the rest of the body falls
# outside the crop, like a webcam that only sees someone's face or torso.
PARTIAL_PROB = 0.3
PARTIAL_SCALE_RANGE = (1.5, 3.0)

HUBER_DELTA = 0.05  # in normalized coords (~5 px at 96x96)
PCK_THRESHOLD = 0.05  # a joint counts as correct within this distance (normalized)
VIS_THRESHOLD = 0.5  # joints scored below this count as not visible
VIS_LOSS_WEIGHT = 0.01  # brings visibility BCE to about the size of the coordinate loss


def load_split(name, limit=None):
    """Return (images, labels, mask) for one split; images get a channel axis."""
    data = np.load(f"{DATA_PREFIX}_{name}.npz")
    images, labels, mask = data["images"], data["labels"], data["mask"]
    if limit:
        images, labels, mask = images[:limit], labels[:limit], mask[:limit]
    return images[..., None], labels, mask


def rotation_matrix(angle):
    return tf.reshape(tf.stack([tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)]),
                      (2, 2))


def warp(image, pts, mask, angle, scale, shift):
    """Rotate, scale, and shift a crop and its (joints, 2) points about the center.

    Joints pushed outside the crop no longer count as labeled.
    """
    # Points move by p' = s * R * (p - c) + c + t, around the crop center c.
    pts = scale * tf.linalg.matvec(rotation_matrix(angle), pts - 0.5) + 0.5 + shift

    # The image transform maps each output pixel back to its input pixel:
    # p = A (p' - c - t) + c, where A = R^-1 / s.
    inv = rotation_matrix(-angle) / scale
    center = tf.constant([IMG_SIZE / 2, IMG_SIZE / 2])
    offset = center - tf.linalg.matvec(inv, center + shift * IMG_SIZE)
    transform = tf.stack([inv[0, 0], inv[0, 1], offset[0],
                          inv[1, 0], inv[1, 1], offset[1], 0.0, 0.0])
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None], transforms=transform[None],
        output_shape=[IMG_SIZE, IMG_SIZE], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="CONSTANT")[0]

    inside = tf.reduce_all((pts >= 0) & (pts <= 1), axis=1)
    return image, pts, mask * tf.cast(inside, tf.float32)


def zoom_on_joint(pts, mask, angle):
    """Return (scale, shift) for warp() that zoom in on a random labeled joint."""
    scale = tf.random.uniform((), *PARTIAL_SCALE_RANGE)
    joint = tf.random.categorical(tf.math.log(mask)[None], 1)[0, 0]
    jitter = tf.random.uniform((2,), -MAX_SHIFT_FRAC, MAX_SHIFT_FRAC)
    # Shift so the chosen joint lands near the center after scaling and rotating.
    focus = tf.gather(pts, joint) - 0.5
    return scale, -scale * tf.linalg.matvec(rotation_matrix(angle), focus) + jitter


def augment(image, label, mask):
    """Randomly flip, rotate, zoom, shift, and relight a crop and its joints."""
    pts = tf.reshape(label, (NUM_JOINTS, 2))

    # Horizontal flip: mirror x and swap each left joint with its right joint.
    flip = tf.random.uniform(()) < 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    flipped = tf.gather(tf.stack([1 - pts[:, 0], pts[:, 1]], axis=1), FLIP_PERM)
    pts = tf.where(flip, flipped, pts)
    mask = tf.where(flip, tf.gather(mask, FLIP_PERM), mask)

    angle = tf.random.uniform((), -1, 1) * MAX_ROTATION_DEG * math.pi / 180
    scale, shift = tf.cond(
        tf.random.uniform(()) < PARTIAL_PROB,
        lambda: zoom_on_joint(pts, mask, angle),
        lambda: (tf.random.uniform((), *SCALE_RANGE),
                 tf.random.uniform((2,), -MAX_SHIFT_FRAC, MAX_SHIFT_FRAC)))
    image, pts, mask = warp(image, pts, mask, angle, scale, shift)

    image = tf.image.random_brightness(image, MAX_BRIGHTNESS_DELTA * 255)
    image = tf.image.random_contrast(image, *CONTRAST_RANGE)
    image = tf.clip_by_value(image, 0, 255)

    return image, tf.reshape(pts, (-1,)), mask


def partial_view(image, label, mask):
    """Zoom in on a random labeled joint with no other changes, for evaluation."""
    pts = tf.reshape(label, (NUM_JOINTS, 2))
    scale, shift = zoom_on_joint(pts, mask, 0.0)
    image, pts, mask = warp(image, pts, mask, 0.0, scale, shift)
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
    """Backbone plus head: raw 0-255 grayscale in, 28 coords then 14 visibility scores out.

    The head flattens a reduced 3x3 feature map instead of global average
    pooling, since pooling would throw away where in the frame features are.
    """
    backbone = build_backbone()
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # MobileNet expects [-1, 1]
    x = backbone(x)
    x = layers.Conv2D(64, 1, activation="relu", name="features")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", name="hidden")(x)
    x = layers.Dropout(0.2)(x)
    coords = layers.Dense(2 * NUM_JOINTS, activation="sigmoid", name="coords")(x)
    visibility = layers.Dense(NUM_JOINTS, activation="sigmoid", name="visibility")(x)
    outputs = layers.Concatenate(name="pose")([coords, visibility])
    return keras.Model(inputs, outputs, name="pose_model"), backbone


def split_target(y_true, y_pred):
    """Return (true points, predicted points, mask) shaped (N, joints, 2 / -)."""
    labels, mask = y_true[:, :2 * NUM_JOINTS], y_true[:, 2 * NUM_JOINTS:]
    return (tf.reshape(labels, (-1, NUM_JOINTS, 2)),
            tf.reshape(y_pred[:, :2 * NUM_JOINTS], (-1, NUM_JOINTS, 2)), mask)


def pose_loss(y_true, y_pred):
    """Huber loss per coordinate over labeled joints, plus visibility cross-entropy."""
    true_pts, pred_pts, mask = split_target(y_true, y_pred)
    err = tf.abs(true_pts - pred_pts)
    huber = tf.where(err < HUBER_DELTA, 0.5 * err ** 2, HUBER_DELTA * (err - 0.5 * HUBER_DELTA))
    per_joint = tf.reduce_sum(huber, axis=2)
    coord_loss = tf.reduce_sum(per_joint * mask) / tf.maximum(tf.reduce_sum(mask), 1)

    vis = y_pred[:, 2 * NUM_JOINTS:]
    vis_loss = tf.reduce_mean(keras.losses.binary_crossentropy(mask[..., None], vis[..., None]))
    return coord_loss + VIS_LOSS_WEIGHT * vis_loss


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


def vis_acc(y_true, y_pred):
    """Fraction of joints whose visible / not-visible call matches the mask."""
    mask, vis = y_true[:, 2 * NUM_JOINTS:], y_pred[:, 2 * NUM_JOINTS:]
    return tf.reduce_mean(tf.cast((vis > VIS_THRESHOLD) == (mask > 0), tf.float32))


def init_from_run(model, run):
    """Copy weights from an earlier run's model wherever layers line up.

    Layers with weights are paired in order. A layer whose weight shapes don't
    match the next unused source layer (such as a newly added head) keeps its
    fresh weights.
    """
    path = os.path.join(MODELS_ROOT, run, MODEL_FILENAME)
    src_layers = [l for l in keras.models.load_model(path, compile=False).layers if l.weights]
    print(f"Initializing from {path}:")
    i = 0
    for layer in model.layers:
        if not layer.weights:
            continue
        if i < len(src_layers) and ([w.shape for w in layer.weights]
                                    == [w.shape for w in src_layers[i].weights]):
            layer.set_weights(src_layers[i].get_weights())
            print(f"  {layer.name} <- {src_layers[i].name}")
            i += 1
        else:
            print(f"  {layer.name}: no matching layer, left as initialized")


def best_val_loss(model_dir):
    """Lowest val loss logged by any earlier epoch, or None before the first one."""
    losses = []
    for path in glob.glob(os.path.join(model_dir, "history_*.csv")):
        with open(path) as f:
            losses += [float(row["val_loss"]) for row in csv.DictReader(f)]
    return min(losses, default=None)


def train_stage(model, train_ds, val_ds, epochs, lr, name, model_dir):
    """Compile and fit for one stage, then load the weights with the best val loss.

    Progress is backed up every BACKUP_EVERY_STEPS batches, so an interrupted
    stage resumes from its last backup. A finished stage is marked with a
    .done file and skipped on later runs.
    """
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    done_path = os.path.join(model_dir, f"stage_{name}.done")
    if os.path.exists(done_path):
        print(f"--- Stage {name} already finished, skipping ---")
        model.load_weights(model_path)
        return

    model.compile(optimizer=keras.optimizers.Adam(lr), loss=pose_loss,
                  metrics=[mean_px_error, pck, vis_acc])
    callbacks = [
        keras.callbacks.BackupAndRestore(os.path.join(model_dir, f"backup_{name}"),
                                         save_freq=BACKUP_EVERY_STEPS),
        # Starting from the best loss so far stops a resumed run from
        # overwriting a better checkpoint saved before the interruption.
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True,
                                        initial_value_threshold=best_val_loss(model_dir)),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.EarlyStopping(patience=6),
        keras.callbacks.CSVLogger(os.path.join(model_dir, f"history_{name}.csv"), append=True),
    ]
    print(f"--- Stage: {name} ({epochs} epochs, lr={lr}) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    model.load_weights(model_path)
    open(done_path, "w").close()


def report(model, images, labels, mask, model_dir, split="val"):
    """Print per-joint error on a split, save a grid of predictions, and return them.

    Also works for models without visibility scores (v1), which output only coords.
    """
    pred = model.predict(images, batch_size=256, verbose=0)
    coords, vis = pred[:, :2 * NUM_JOINTS], pred[:, 2 * NUM_JOINTS:]
    has_vis = vis.shape[1] > 0
    dist = np.linalg.norm((coords - labels).reshape(-1, NUM_JOINTS, 2), axis=2)

    labeled = dist[mask > 0]
    vis_text = (f", visibility acc {((vis > VIS_THRESHOLD) == (mask > 0)).mean():.3f}"
                if has_vis else "")
    print(f"Overall {split} error: {IMG_SIZE * labeled.mean():.2f} px, "
          f"PCK {(labeled < PCK_THRESHOLD).mean():.3f}{vis_text}")
    print(f"Per-joint {split} error (px at 96x96) / PCK:")
    for j, name in enumerate(JOINT_ORDER):
        d = dist[mask[:, j] > 0, j]
        print(f"  {name:<11} {IMG_SIZE * d.mean():5.2f} px   {(d < PCK_THRESHOLD).mean():.3f}")

    # True joints in green, predictions in red (hollow if scored not visible),
    # every crop scaled up 2x.
    tile, cols, rows = 2 * IMG_SIZE, 6, 4
    grid = Image.new("RGB", (cols * tile, rows * tile))
    for k, i in enumerate(np.random.RandomState(0).choice(len(images), cols * rows, replace=False)):
        crop = Image.fromarray(images[i, ..., 0]).convert("RGB").resize((tile, tile))
        draw = ImageDraw.Draw(crop)
        for j in range(NUM_JOINTS):
            px, py = coords[i, 2 * j] * tile, coords[i, 2 * j + 1] * tile
            if mask[i, j]:
                tx, ty = labels[i, 2 * j] * tile, labels[i, 2 * j + 1] * tile
                draw.line((tx, ty, px, py), fill=(255, 255, 0))
                draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=(0, 255, 0))
            visible = not has_vis or vis[i, j] > VIS_THRESHOLD
            draw.ellipse((px - 3, py - 3, px + 3, py + 3),
                         fill=(255, 0, 0) if visible else None, outline=(255, 0, 0))
        grid.paste(crop, ((k % cols) * tile, (k // cols) * tile))
    path = os.path.join(model_dir, f"{split}_predictions.png")
    grid.save(path)
    print(f"Saved prediction grid to {path}")
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run's folder under models/")
    parser.add_argument("--init-from", metavar="RUN",
                        help="start from an earlier run's weights instead of ImageNet")
    parser.add_argument("--limit", type=int, help="only use the first N crops of each split (for testing)")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--full-epochs", type=int, default=40)
    args = parser.parse_args()
    model_dir = os.path.join(MODELS_ROOT, args.run)
    os.makedirs(model_dir, exist_ok=True)
    script_copy = os.path.join(model_dir, "train.py")
    if not os.path.exists(script_copy):  # keep the code that started the run
        shutil.copy(__file__, script_copy)

    train = load_split("train", args.limit)
    val = load_split("val", args.limit)
    train_ds, val_ds = make_dataset(*train, training=True), make_dataset(*val, training=False)
    print(f"Train: {len(train[0])} crops, val: {len(val[0])} crops")

    model, backbone = build_model()
    if args.init_from:
        init_from_run(model, args.init_from)
    model.summary()

    backbone.trainable = False
    train_stage(model, train_ds, val_ds, args.head_epochs, 1e-3, "head", model_dir)
    backbone.trainable = True
    train_stage(model, train_ds, val_ds, args.full_epochs, 1e-4, "full", model_dir)

    report(model, *val, model_dir)


if __name__ == "__main__":
    main()
