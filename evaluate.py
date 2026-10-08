"""
Evaluate one or more trained runs on a preprocessed split and compare them.

For each run, prints per-joint error and saves a grid of predictions to
models/<run>/{split}_predictions.png. With several runs, also prints a
side-by-side comparison table.

--partial zooms every crop in on one random labeled joint first (the same
zoom for every run), to test crops that only show part of a body, such as a
webcam pointed at someone's face.

Usage: python evaluate.py [RUN ...] [--split test|val|train] [--partial] [--limit N]
  e.g. python evaluate.py v1 v2 --split test --partial
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from train import (DEFAULT_RUN, IMG_SIZE, JOINT_ORDER, MODEL_FILENAME, MODELS_ROOT,
                   NUM_JOINTS, PCK_THRESHOLD, VIS_THRESHOLD, load_split, partial_view, report)

PARTIAL_SEED = 0


def make_partial(images, labels, mask):
    """Return copies of the crops, each zoomed in on one random labeled joint."""
    tf.random.set_seed(PARTIAL_SEED)
    ds = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), labels, mask))
    ds = ds.map(partial_view).batch(1024)
    parts = [np.concatenate(x) for x in zip(*[(i.numpy(), l.numpy(), m.numpy()) for i, l, m in ds])]
    return np.clip(np.round(parts[0]), 0, 255).astype(np.uint8), parts[1], parts[2]


def joint_stats(pred, labels, mask):
    """Return {row name: (mean px error, PCK)} over labeled joints, plus visibility acc."""
    coords, vis = pred[:, :2 * NUM_JOINTS], pred[:, 2 * NUM_JOINTS:]
    dist = np.linalg.norm((coords - labels).reshape(-1, NUM_JOINTS, 2), axis=2)
    stats = {}
    for j, name in enumerate(JOINT_ORDER):
        d = dist[mask[:, j] > 0, j]
        stats[name] = (IMG_SIZE * d.mean(), (d < PCK_THRESHOLD).mean())
    d = dist[mask > 0]
    stats["overall"] = (IMG_SIZE * d.mean(), (d < PCK_THRESHOLD).mean())
    stats["vis acc"] = ((vis > VIS_THRESHOLD) == (mask > 0)).mean() if vis.shape[1] else None
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=[DEFAULT_RUN],
                        help=f"run folders under {MODELS_ROOT}/ (default: {DEFAULT_RUN})")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--partial", action="store_true",
                        help="zoom each crop in on one joint to test partial bodies")
    parser.add_argument("--limit", type=int, help="only use the first N crops of the split")
    args = parser.parse_args()

    images, labels, mask = load_split(args.split, args.limit)
    name = args.split
    if args.partial:
        images, labels, mask = make_partial(images, labels, mask)
        name += "_partial"
    print(f"{name}: {len(images)} crops, {mask.mean():.0%} of joints visible and labeled")

    results = {}
    for run in args.runs:
        model_dir = os.path.join(MODELS_ROOT, run)
        model = keras.models.load_model(os.path.join(model_dir, MODEL_FILENAME), compile=False)
        print(f"\n=== {run} ===")
        pred = report(model, images, labels, mask, model_dir, name)
        results[run] = joint_stats(pred, labels, mask)

    if len(args.runs) > 1:
        print(f"\nComparison on {name} (px error / PCK):")
        print(f"  {'':<11}" + "".join(f"{run:>18}" for run in args.runs))
        for row in ["overall"] + JOINT_ORDER:
            cells = "".join(f"{results[r][row][0]:10.2f} / {results[r][row][1]:.3f}"
                            for r in args.runs)
            print(f"  {row:<11}{cells}")
        acc = [results[r]["vis acc"] for r in args.runs]
        print(f"  {'vis acc':<11}" + "".join(f"{'n/a' if a is None else f'{a:.3f}':>18}"
                                            for a in acc))


if __name__ == "__main__":
    main()
