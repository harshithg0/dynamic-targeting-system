# Pose model versions

Each folder is one training run of `train.py`: the trained model
(`pose_model.keras`), the per-epoch training history, and a copy of the
`train.py` that trained it. All three use the same small model: a
MobileNetV1-0.25 backbone taking a 96x96 grayscale crop of one person, with
outputs for 14 joints (head, neck, shoulders, elbows, wrists, hips, knees,
ankles).

Compare runs with `python evaluate.py v1 v2 v3 --split test [--partial]`, and
try one on the webcam with `python live_demo.py v3`.

## v1: joint coordinates only

- Outputs x, y for every joint. Trained from ImageNet weights on whole-body
  crops from COCO and MPII. Code: git tag `v1`.
- Works on whole-body crops, but it always places all 14 joints. With only a
  face in view, it squeezes a whole body into the frame.

## v2: adds a visibility score per joint

- Adds an output scoring whether each joint is actually in the crop, so off-screen
  joints can be hidden. 30% of training crops were zoomed in 1.5-3x on one joint
  to show partial bodies. Started from v1's weights. Code: commit `46b93bb`.
- Did not fix the webcam. The zoomed crops were cut out of the 96x96 crops, so
  every partial-body example was blurry. The model learned "blurry means part of
  a body" instead of checking for legs. On sharp crops, like a webcam's, it still
  showed the whole body. Its visibility accuracy on sharp partial crops (57%) is
  worse than calling every joint not visible (62%).

## v3: sharp partial-body crops

- Same model as v2, but the partial-body crops are cut from the full-resolution
  photos (`preprocess_dataset.py --partial`), so they're as sharp as a real
  close-up. Half of them center on the head or neck. Training mixes them with
  the whole-body crops; the blurry zoom augmentation is gone. Started from v2's
  weights. Code: commit `d662036`.
- Hides off-screen joints on sharp crops. It scores about the same on sharp and
  blurred crops, so it no longer relies on blur. Confirmed working on the
  webcam (face-only view).
- The model here is from epoch 28 of the second training stage. Training was
  still running when this was committed, but later epochs were improving by
  about 0.01 px, so a newer file would barely differ.

## Results on the test split

Average error in pixels on the 96x96 crop / fraction of joints within 5% of the
crop size (PCK@0.05). Visibility is the fraction of joints correctly called in
or out of the crop.

| Test crops | v1 | v2 | v3 |
|---|---|---|---|
| Whole body | 10.28 px / 0.276 | 10.31 px / 0.275 | 10.22 px / 0.280 |
| Whole body, visibility | n/a | 0.900 | 0.889 |
| Partial body, sharp | 29.25 px / 0.027 | 27.28 px / 0.036 | 18.62 px / 0.109 |
| Partial body, sharp, visibility | n/a | **0.568** | **0.804** |
| Partial body, blurred, visibility | n/a | 0.785 | 0.794 |

Sharp partial crops come from `preprocess_dataset.py --partial`, and
`evaluate.py --blurred` blurs them the way v2's training crops were blurred.

## Known limits and next steps

- Wrists are the weakest joint: about 15 px off on whole-body crops and
  25 px on partial ones.
- Partial bodies are still much harder than whole ones (18.6 px vs 10.2 px).
- The model expects a crop around one person. Following a chosen person
  around the camera frame needs a person detector and tracker in front of it.
- Bigger accuracy gains likely need heatmap outputs, or a larger backbone or
  input size.
