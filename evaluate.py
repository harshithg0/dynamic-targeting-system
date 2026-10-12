"""
Evaluate one or more trained runs on a preprocessed split and compare them.

For each run, prints per-joint error and saves a grid of predictions to
models/<run>/{split}_predictions.png. With several runs, also prints a
side-by-side comparison table.

--partial uses the partial-body crops from preprocess_dataset.py --partial
instead, which show only part of a person, like a webcam pointed at someone's
face. Adding --blurred blurs those crops to the detail they would keep if cut
out of a 96x96 whole-body crop, which is how v2's partial crops were made.
A model that scores much better blurred than sharp is using blur as a cue.

Usage: python evaluate.py [RUN ...] [--split test|val|train] [--partial [--blurred]]
                          [--limit N]
  e.g. python evaluate.py v2 v3 --split test --partial
"""
import argparse
import os

import cv2
import numpy as np
from tensorflow import keras

from train import (DATA_PREFIX, DEFAULT_RUN, IMG_SIZE, JOINT_ORDER, MODEL_FILENAME,
                   MODELS_ROOT, NUM_JOINTS, PCK_THRESHOLD, VIS_THRESHOLD, load_split, report)


def blur_like_zoomed(images, zooms):
    """Shrink each crop by its zoom factor and scale it back up."""
    out = np.empty_like(images)
    for i, (im, zoom) in enumerate(zip(images[..., 0], zooms)):
        small = max(4, round(IMG_SIZE / zoom))
        small_im = cv2.resize(im, (small, small), interpolation=cv2.INTER_AREA)
        out[i, ..., 0] = cv2.resize(small_im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return out


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
                        help="use the partial-body crops to test partial bodies")
    parser.add_argument("--blurred", action="store_true",
                        help="with --partial, blur the crops like v2's zoomed training crops")
    parser.add_argument("--limit", type=int, help="only use the first N crops of the split")
    args = parser.parse_args()

    if args.blurred and not args.partial:
        parser.error("--blurred needs --partial")
    images, labels, mask = load_split(args.split, args.limit, partial=args.partial)
    name = args.split + ("_partial" if args.partial else "")
    if args.blurred:
        zooms = np.load(f"{DATA_PREFIX}_partial_{args.split}.npz")["zoom"][:len(images)]
        images = blur_like_zoomed(images, zooms)
        name += "_blurred"
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
