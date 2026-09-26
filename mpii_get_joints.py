"""
Extract 14-joint 2D pose labels from MPII Human Pose annotations.

Reads the MPII annotation .mat file, keeps annotated people who have all
required joints present, and writes {filepath, joints} records to
OUTPUT_PATH.
"""
import json
import os

import numpy as np
import scipy.io as sio

MAT_PATH = "mpii_human_pose_v1/mpii_human_pose_v1_u12_1.mat"
IMG_DIR = "mpii_human_pose_v1/images"
OUTPUT_PATH = "mpii_13joint_labels.json"

MPII_JOINTS = {
    0: "r_ankle", 1: "r_knee", 2: "r_hip", 3: "l_hip", 4: "l_knee", 5: "l_ankle",
    6: "pelvis", 7: "thorax", 8: "upper_neck", 9: "head_top",
    10: "r_wrist", 11: "r_elbow", 12: "r_shoulder", 13: "l_shoulder", 14: "l_elbow", 15: "l_wrist",
}
REQUIRED = [
    "head_top", "upper_neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def rect_joints(rect):
    """Return {mpii joint name: [x, y]} for one annotated person's keypoints."""
    points = rect.annopoints.point
    point_list = points if isinstance(points, np.ndarray) else [points]

    joints = {}
    for p in point_list:
        name = MPII_JOINTS.get(int(p.id))
        if name:
            joints[name] = [float(p.x), float(p.y)]
    return joints


def main():
    mat = sio.loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)
    release = mat["RELEASE"]
    is_train = release.img_train

    records = []
    skipped_missing = 0
    skipped_incomplete = 0

    for idx, anno in enumerate(release.annolist):
        if not is_train[idx]:
            continue

        img_path = os.path.join(IMG_DIR, anno.image.name)
        if not os.path.exists(img_path):
            skipped_missing += 1
            continue

        annorect = anno.annorect
        rects = annorect if isinstance(annorect, np.ndarray) else [annorect]

        for rect in rects:
            if not hasattr(rect, "annopoints") or not hasattr(rect.annopoints, "point"):
                continue

            joints = rect_joints(rect)
            if not all(r in joints for r in REQUIRED):
                skipped_incomplete += 1
                continue

            records.append({
                "filepath": img_path,
                "joints": {
                    "head": joints["head_top"], "neck": joints["upper_neck"],
                    "l_shoulder": joints["l_shoulder"], "r_shoulder": joints["r_shoulder"],
                    "l_elbow": joints["l_elbow"], "r_elbow": joints["r_elbow"],
                    "l_wrist": joints["l_wrist"], "r_wrist": joints["r_wrist"],
                    "l_hip": joints["l_hip"], "r_hip": joints["r_hip"],
                    "l_knee": joints["l_knee"], "r_knee": joints["r_knee"],
                    "l_ankle": joints["l_ankle"], "r_ankle": joints["r_ankle"],
                },
            })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f)

    print(f"Extracted {len(records)} records, skipped {skipped_missing} missing images, "
          f"{skipped_incomplete} incomplete annotations")


if __name__ == "__main__":
    main()
