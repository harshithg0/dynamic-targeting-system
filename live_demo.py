"""
Run the trained pose model on a live webcam feed.

The model was trained on square crops framed around one person, so it only
looks at a square box in the middle of the frame. Stand so your body fills
the box, head near the top and feet near the bottom.

Controls: '+' / '-' = grow / shrink the box, 'q' = quit

Usage: python live_demo.py [RUN]   (a folder under models/, default v1)
"""
import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from train import DEFAULT_RUN, IMG_SIZE, JOINT_ORDER, MODEL_FILENAME, MODELS_ROOT

CAMERA_INDEX = 0
MIRROR_CAMERA = True
WINDOW_NAME = "Pose Model Demo"

BOX_FRAC = 0.95       # starting box side, as a fraction of the frame height
BOX_STEP = 0.05
MIN_BOX_FRAC = 0.3

BOX_COLOR = (255, 255, 0)     # cyan, BGR
JOINT_COLOR = (0, 0, 255)     # red
BONE_COLOR = (0, 255, 0)      # green
JOINT_RADIUS = 5

BONES = [
    ("head", "neck"),
    ("neck", "l_shoulder"), ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default=DEFAULT_RUN, help="run folder under models/")
    args = parser.parse_args()

    model_path = os.path.join(MODELS_ROOT, args.run, MODEL_FILENAME)
    model = keras.models.load_model(model_path, compile=False)
    # Compiling the call to a graph makes each frame ~10x faster than eager mode.
    predict = tf.function(lambda x: model(x, training=False))
    cap = cv2.VideoCapture(CAMERA_INDEX)
    box_frac = BOX_FRAC
    print("Controls: '+' / '-' = grow / shrink the box, 'q' = quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        side = int(min(h, w) * box_frac)
        x0, y0 = (w - side) // 2, (h - side) // 2

        crop = frame[y0:y0 + side, x0:x0 + side]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        start = time.perf_counter()
        pred = predict(small[None, ..., None].astype(np.float32)).numpy()[0]
        infer_ms = 1000 * (time.perf_counter() - start)

        # Predictions are in [0, 1] within the crop; map them back to the frame.
        pts = {name: (int(x0 + pred[2 * j] * side), int(y0 + pred[2 * j + 1] * side))
               for j, name in enumerate(JOINT_ORDER)}

        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), BOX_COLOR, 2)
        for a, b in BONES:
            cv2.line(frame, pts[a], pts[b], BONE_COLOR, 2)
        for p in pts.values():
            cv2.circle(frame, p, JOINT_RADIUS, JOINT_COLOR, -1)
        cv2.putText(frame, f"{args.run}: {infer_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            box_frac = min(1.0, box_frac + BOX_STEP)
        elif key == ord("-"):
            box_frac = max(MIN_BOX_FRAC, box_frac - BOX_STEP)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
