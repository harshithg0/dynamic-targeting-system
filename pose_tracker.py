import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ============================
# Global Variables
# ============================

# Model
MODEL_PATH = "pose_landmarker.task"

# Detection settings
NUM_POSES = 1                     # how many people to detect at once
MIN_POSE_DETECTION_CONFIDENCE = 0.5   # confidence a person exists in frame
MIN_POSE_PRESENCE_CONFIDENCE = 0.5    # confidence pose is still present between frames
MIN_TRACKING_CONFIDENCE = 0.5         # confidence when tracking landmarks frame-to-frame

# Which body part to highlight (see PoseLandmark enum for full list)
TARGET_LANDMARK_NAME = "RIGHT_WRIST"

# Camera
CAMERA_INDEX = 0        # 0 = default laptop camera
MIRROR_CAMERA = True    # flip horizontally for natural mirror view

# Display
ALL_LANDMARKS_COLOR = (0, 255, 0)   # green, BGR format
ALL_LANDMARKS_RADIUS = 4
TARGET_COLOR = (0, 0, 255)          # red, BGR format
TARGET_RADIUS = 10
TARGET_THICKNESS = 2
WINDOW_NAME = "Pose Tracking"
QUIT_KEY = 'q'

# ============================
# Setup PoseLandmarker
# ============================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

TARGET_LANDMARK = mp.solutions.pose.PoseLandmark[TARGET_LANDMARK_NAME].value

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses=NUM_POSES,
    min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
    min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    result_callback=result_callback
)

landmarker = PoseLandmarker.create_from_options(options)

# ============================
# Camera Loop
# ============================

cap = cv2.VideoCapture(CAMERA_INDEX)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if MIRROR_CAMERA:
        frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int((time.time() - start_time) * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    h, w, _ = frame.shape

    if latest_result and latest_result.pose_landmarks:
        for pose_landmarks in latest_result.pose_landmarks:
            for lm in pose_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), ALL_LANDMARKS_RADIUS, ALL_LANDMARKS_COLOR, -1)

            target = pose_landmarks[TARGET_LANDMARK]
            tx, ty = int(target.x * w), int(target.y * h)
            cv2.circle(frame, (tx, ty), TARGET_RADIUS, TARGET_COLOR, TARGET_THICKNESS)
            print(f"Target pixel: ({tx}, {ty})  norm: ({target.x:.3f}, {target.y:.3f})")

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()