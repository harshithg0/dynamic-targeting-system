"""
Evaluate one or more trained runs on a preprocessed split and compare them.

For each run, prints per-joint error and saves a grid of predictions to
models/<run>/{split}_predictions.png. With several runs, also prints a
side-by-side comparison table.

Usage: python evaluate.py [RUN ...] [--split test|val|train] [--limit N]
  e.g. python evaluate.py v1 v2 --split test
"""
import argparse
import os

import numpy as np
from tensorflow import keras

from train import (DEFAULT_RUN, IMG_SIZE, JOINT_ORDER, MODEL_FILENAME, MODELS_ROOT,
                   NUM_JOINTS, PCK_THRESHOLD, load_split, report)


def joint_stats(pred, labels, mask):
    """Return {joint or 'overall': (mean px error, PCK)} over labeled joints."""
    dist = np.linalg.norm((pred - labels).reshape(-1, NUM_JOINTS, 2), axis=2)
    stats = {}
    for j, name in enumerate(JOINT_ORDER):
        d = dist[mask[:, j] > 0, j]
        stats[name] = (IMG_SIZE * d.mean(), (d < PCK_THRESHOLD).mean())
    d = dist[mask > 0]
    stats["overall"] = (IMG_SIZE * d.mean(), (d < PCK_THRESHOLD).mean())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=[DEFAULT_RUN],
                        help=f"run folders under {MODELS_ROOT}/ (default: {DEFAULT_RUN})")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, help="only use the first N crops of the split")
    args = parser.parse_args()

    images, labels, mask = load_split(args.split, args.limit)
    print(f"{args.split}: {len(images)} crops")

    results = {}
    for run in args.runs:
        model_dir = os.path.join(MODELS_ROOT, run)
        model = keras.models.load_model(os.path.join(model_dir, MODEL_FILENAME), compile=False)
        print(f"\n=== {run} ===")
        pred = report(model, images, labels, mask, model_dir, args.split)
        results[run] = joint_stats(pred, labels, mask)

    if len(args.runs) > 1:
        print(f"\nComparison on {args.split} (px error / PCK):")
        print(f"  {'':<11}" + "".join(f"{run:>18}" for run in args.runs))
        for name in ["overall"] + JOINT_ORDER:
            cells = "".join(f"{results[r][name][0]:10.2f} / {results[r][name][1]:.3f}"
                            for r in args.runs)
            print(f"  {name:<11}{cells}")


if __name__ == "__main__":
    main()
