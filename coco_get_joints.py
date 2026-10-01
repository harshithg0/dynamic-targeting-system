"""
Extract 14-joint 2D pose labels from COCO person-keypoints annotations.

For each annotated person with at least MIN_KEYPOINTS labeled keypoints,
derives a head/neck/shoulders/elbows/wrists/hips/knees/ankles joint set and
writes {filepath, joints} records to OUTPUT_PATH. Joints COCO marks as
unlabeled (v=0) are written as null instead of COCO's placeholder (0, 0).
"""
import json
import os
from pycocotools.coco import COCO

ANN_PATH = "coco-2017/raw/person_keypoints_train2017.json"
IMG_DIR = "coco-2017/train/data"
OUTPUT_PATH = "coco_13joint_labels.json"
MIN_KEYPOINTS = 8

# COCO keypoint order (https://cocodataset.org/#keypoints-eval):
# 0 nose, 5/6 l/r shoulder, 7/8 l/r elbow, 9/10 l/r wrist,
# 11/12 l/r hip, 13/14 l/r knee, 15/16 l/r ankle
# Each point is (x, y, v): v=0 unlabeled, v=1 labeled but occluded,
# v=2 labeled and visible.


def extract_joints(keypoints):
    """Convert a flat COCO keypoint list into our 14-joint dict."""
    triples = [keypoints[i:i + 3] for i in range(0, len(keypoints), 3)]
    # Unlabeled points are None so they can't be mistaken for real (0, 0) coords.
    pts = [[x, y] if v > 0 else None for x, y, v in triples]

    l_sh, r_sh = pts[5], pts[6]
    neck = [(l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2] if l_sh and r_sh else None

    return {
        "head": pts[0], "neck": neck,
        "l_shoulder": l_sh, "r_shoulder": r_sh,
        "l_elbow": pts[7], "r_elbow": pts[8],
        "l_wrist": pts[9], "r_wrist": pts[10],
        "l_hip": pts[11], "r_hip": pts[12],
        "l_knee": pts[13], "r_knee": pts[14],
        "l_ankle": pts[15], "r_ankle": pts[16],
    }


def main():
    coco = COCO(ANN_PATH)
    person_cat_id = coco.getCatIds(catNms=["person"])[0]
    img_ids = coco.getImgIds(catIds=[person_cat_id])

    records = []
    skipped_missing = 0

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
        anns = [a for a in coco.loadAnns(ann_ids) if a["num_keypoints"] >= MIN_KEYPOINTS]
        if not anns:
            continue

        info = coco.loadImgs(img_id)[0]
        local_path = os.path.join(IMG_DIR, info["file_name"])
        if not os.path.exists(local_path):
            skipped_missing += 1
            continue

        for ann in anns:
            records.append({
                "filepath": local_path,
                "joints": extract_joints(ann["keypoints"]),
            })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f)

    print(f"Total person images: {len(img_ids)}")
    print(f"Extracted {len(records)} person records, skipped {skipped_missing} images missing locally")


if __name__ == "__main__":
    main()
