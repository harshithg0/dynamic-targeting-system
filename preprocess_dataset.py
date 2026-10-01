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

Usage: python preprocess_dataset.py [--limit N]
"""
import argparse
import json
import random
from collections import defaultdict
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


def process_image(item):
    """Crop, grayscale, and resize every person record from one source image."""
    path, records = item
    try:
        img = Image.open(path).convert("L")
    except OSError:
        return []

    results = []
    for r in records:
        points = [r["joints"][j] for j in JOINT_ORDER]
        x0, y0, side = compute_crop_box([p for p in points if p is not None])
        if side < MIN_CROP_SIDE:
            continue

        crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((TARGET_SIZE, TARGET_SIZE))

        label = np.zeros(2 * len(JOINT_ORDER), dtype=np.float32)
        mask = np.zeros(len(JOINT_ORDER), dtype=np.float32)
        for i, p in enumerate(points):
            if p is not None:
                label[2 * i] = (p[0] - x0) / side
                label[2 * i + 1] = (p[1] - y0) / side
                mask[i] = 1

        results.append((np.asarray(crop, dtype=np.uint8), label, mask, path))
    return results


def build_split(name, items, pool, prefix):
    """Process one split's images in parallel and save the crops to an .npz file."""
    n_max = sum(len(records) for _, records in items)
    images = np.zeros((n_max, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    labels = np.zeros((n_max, 2 * len(JOINT_ORDER)), dtype=np.float32)
    masks = np.zeros((n_max, len(JOINT_ORDER)), dtype=np.float32)
    sources = []

    n = 0
    for i, results in enumerate(pool.imap(process_image, items, chunksize=64)):
        for img, label, mask, path in results:
            images[n], labels[n], masks[n] = img, label, mask
            sources.append(path)
            n += 1
        if (i + 1) % 5000 == 0:
            print(f"  {name}: {i + 1}/{len(items)} images")

    np.savez(f"{prefix}_{name}.npz", images=images[:n], labels=labels[:n],
             mask=masks[:n], source=np.array(sources))
    print(f"{name}: {n} crops from {len(items)} images ({n_max - n} skipped)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int,
                        help="only process the first N images of each split, saved "
                             f"as {OUTPUT_PREFIX}_sample_*.npz (for testing)")
    args = parser.parse_args()
    prefix = f"{OUTPUT_PREFIX}_sample" if args.limit else OUTPUT_PREFIX

    records = load_records(LABEL_PATHS)
    print(f"{len(records)} records with at least {MIN_LABELED_JOINTS} labeled joints")

    splits = split_by_image(records)
    with Pool() as pool:
        for name, items in splits.items():
            build_split(name, items[:args.limit], pool, prefix)


if __name__ == "__main__":
    main()
