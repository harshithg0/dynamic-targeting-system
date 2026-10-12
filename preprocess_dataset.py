"""
Build 96x96 grayscale training crops from the extracted COCO/MPII joint labels.

For each person with enough labeled joints, crops a padded square around the
labeled joints, converts it to grayscale, resizes it to TARGET_SIZE, and
normalizes the joint coordinates to [0, 1] within the crop. People are split
into train/val/test by source image, so people from the same photo never end
up in different splits. Each split is saved as {prefix}_{split}.npz with:
  images  uint8   (N, 96, 96)
  labels  float32 (N, 28)  x, y per joint in JOINT_ORDER; 0 where unlabeled
  mask    float32 (N, 14)  1 if the joint is labeled, else 0
  source  str     (N,)     source image path, for debugging

With --partial, instead builds crops that show only part of each person:
a smaller square, cut from the full-resolution photo, around one labeled
joint, so the rest of the body falls outside the crop (like a webcam that
only sees someone's face). Joints outside the crop are unlabeled in mask.
These are saved as {prefix}_partial_{split}.npz with one more array:
  zoom    float32 (N,)     how much smaller the crop is than the whole-body crop

Cutting these from the full-resolution photo keeps them as sharp as a real
close-up. Zooming into the 96x96 whole-body crops instead makes them blurry,
and a model trained that way learns to treat blur as a sign of a partial
body.

Usage: python preprocess_dataset.py [--partial] [--limit N]
"""
import argparse
import json
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image

LABEL_PATHS = ["coco_13joint_labels.json", "mpii_13joint_labels.json"]
OUTPUT_PREFIX = "pose_dataset"

JOINT_ORDER = [
    "head", "neck",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
]

TARGET_SIZE = 96
PADDING_FRAC = 0.2      # margin on each side, as a fraction of the box's longer side
MIN_LABELED_JOINTS = 8  # skip people with fewer labeled joints than this
MIN_CROP_SIDE = 32      # skip crops smaller than this in the source image, in px
SPLIT_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Partial crops (--partial)
PARTIAL_ZOOM_RANGE = (1.5, 4.0)  # whole-body crop side / partial crop side
MIN_PARTIAL_SIDE = 64      # partial crops smaller than this in the photo would be blurry
HEAD_FOCUS_PROB = 0.5      # how often to center on the head or neck (face-only views)
PARTIAL_SHIFT_FRAC = 0.15  # random offset of the focus joint, as a fraction of the side
SEED = 42


def load_records(paths):
    """Load label records, keeping people with at least MIN_LABELED_JOINTS joints."""
    records = []
    for path in paths:
        with open(path) as f:
            records.extend(json.load(f))
    return [r for r in records
            if sum(r["joints"][j] is not None for j in JOINT_ORDER) >= MIN_LABELED_JOINTS]


def split_by_image(records):
    """Group records by source image and assign whole images to each split."""
    by_image = defaultdict(list)
    for r in records:
        by_image[r["filepath"]].append(r)

    paths = sorted(by_image)
    random.Random(SEED).shuffle(paths)

    splits, start = {}, 0
    for i, (name, frac) in enumerate(SPLIT_FRACS.items()):
        end = len(paths) if i == len(SPLIT_FRACS) - 1 else start + int(len(paths) * frac)
        splits[name] = [(p, by_image[p]) for p in paths[start:end]]
        start = end
    return splits


def compute_crop_box(points):
    """Return (x0, y0, side) of a padded square around the given points.

    The square may extend past the image edges; PIL fills that area with
    black. Keeping the crop square means resizing to TARGET_SIZE never
    stretches the body's proportions.
    """
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * PADDING_FRAC)
    return round(cx - side / 2), round(cy - side / 2), round(side)


def compute_partial_box(points, rng):
    """Return (x0, y0, side, zoom) of a smaller square around one labeled joint, or None.

    The square is 1/zoom the side of the whole-body crop, centered near a
    random labeled joint (often the head or neck). Returns None when the
    person is too small in the photo for a sharp crop at the smallest zoom.
    """
    labeled = [p for p in points if p is not None]
    _, _, body_side = compute_crop_box(labeled)
    max_zoom = min(PARTIAL_ZOOM_RANGE[1], body_side / MIN_PARTIAL_SIDE)
    if max_zoom < PARTIAL_ZOOM_RANGE[0]:
        return None
    zoom = rng.uniform(PARTIAL_ZOOM_RANGE[0], max_zoom)
    side = body_side / zoom

    head_or_neck = [p for p in points[:2] if p is not None]
    focus = rng.choice(head_or_neck if head_or_neck and rng.random() < HEAD_FOCUS_PROB
                       else labeled)
    cx = focus[0] + rng.uniform(-1, 1) * PARTIAL_SHIFT_FRAC * side
    cy = focus[1] + rng.uniform(-1, 1) * PARTIAL_SHIFT_FRAC * side
    return round(cx - side / 2), round(cy - side / 2), round(side), zoom


def process_image(item, partial_crops=False):
    """Crop, grayscale, and resize every person record from one source image.

    Returns (image, label, mask, path, zoom) per kept person; zoom is 1 for
    whole-body crops.
    """
    path, records = item
    try:
        img = Image.open(path).convert("L")
    except OSError:
        return []
    rng = random.Random(f"{SEED}:{path}")  # same crops on every run

    results = []
    for r in records:
        points = [r["joints"][j] for j in JOINT_ORDER]
        if partial_crops:
            box = compute_partial_box(points, rng)
            if box is None:
                continue
            x0, y0, side, zoom = box
        else:
            x0, y0, side = compute_crop_box([p for p in points if p is not None])
            zoom = 1.0
            if side < MIN_CROP_SIDE:
                continue

        crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((TARGET_SIZE, TARGET_SIZE))

        label = np.zeros(2 * len(JOINT_ORDER), dtype=np.float32)
        mask = np.zeros(len(JOINT_ORDER), dtype=np.float32)
        for i, p in enumerate(points):
            if p is None:
                continue
            x, y = (p[0] - x0) / side, (p[1] - y0) / side
            if 0 <= x <= 1 and 0 <= y <= 1:  # always true for whole-body crops
                label[2 * i], label[2 * i + 1] = x, y
                mask[i] = 1

        results.append((np.asarray(crop, dtype=np.uint8), label, mask, path, zoom))
    return results


def build_split(name, items, pool, prefix, partial_crops):
    """Process one split's images in parallel and save the crops to an .npz file."""
    n_max = sum(len(records) for _, records in items)
    images = np.zeros((n_max, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    labels = np.zeros((n_max, 2 * len(JOINT_ORDER)), dtype=np.float32)
    masks = np.zeros((n_max, len(JOINT_ORDER)), dtype=np.float32)
    zooms = np.zeros(n_max, dtype=np.float32)
    sources = []

    n = 0
    work = partial(process_image, partial_crops=partial_crops)
    for i, results in enumerate(pool.imap(work, items, chunksize=64)):
        for img, label, mask, path, zoom in results:
            images[n], labels[n], masks[n], zooms[n] = img, label, mask, zoom
            sources.append(path)
            n += 1
        if (i + 1) % 5000 == 0:
            print(f"  {name}: {i + 1}/{len(items)} images")

    arrays = dict(images=images[:n], labels=labels[:n], mask=masks[:n], source=np.array(sources))
    if partial_crops:
        arrays["zoom"] = zooms[:n]
    np.savez(f"{prefix}_{name}.npz", **arrays)
    print(f"{name}: {n} crops from {len(items)} images ({n_max - n} skipped)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true",
                        help=f"build partial-body crops, saved as {OUTPUT_PREFIX}_partial_*.npz")
    parser.add_argument("--limit", type=int,
                        help="only process the first N images of each split, saved "
                             f"as {OUTPUT_PREFIX}_sample_*.npz (for testing)")
    args = parser.parse_args()
    prefix = OUTPUT_PREFIX + ("_partial" if args.partial else "") + ("_sample" if args.limit else "")

    records = load_records(LABEL_PATHS)
    print(f"{len(records)} records with at least {MIN_LABELED_JOINTS} labeled joints")

    splits = split_by_image(records)
    with Pool() as pool:
        for name, items in splits.items():
            build_split(name, items[:args.limit], pool, prefix, args.partial)


if __name__ == "__main__":
    main()
