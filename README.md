# Laser Tracker

A computer vision project that uses [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) to detect body landmarks from a live webcam feed, with a pan-tilt servo mount equipped with a laser.

## How it works

1. Captures live video from a webcam using OpenCV.
2. Runs each frame through MediaPipe's Pose Landmarker model to detect 33 body landmarks (shoulders, wrists, hips, knees, etc.).
3. Highlights a configurable target landmark and prints its pixel coordinates.
4. Converts the target's position into pan/tilt servo angles and sends them to a microcontroller to physically aim a laser at the tracked point.
