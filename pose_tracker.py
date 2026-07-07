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

# Which body part to highlight
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
TYPE_TARGET_KEY = 't'   # enter typing mode to name a new target landmark
INPUT_COLOR = (255, 255, 0)   # cyan, BGR format
ERROR_COLOR = (0, 0, 255)     # red, BGR format
STATUS_DISPLAY_FRAMES = 60    # how many frames to show the error message

# ============================
# Setup PoseLandmarker
# ============================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Basic points only: body joints plus a simple face outline (skips
# fine-grained face, hand, and foot landmarks). Values are each name's real
# index in MediaPipe's 33-point landmark output.
POSE_LANDMARKS = {
    "NOSE": 0, "LEFT_EYE": 2, "RIGHT_EYE": 5, "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12, "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16, "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26, "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}
INDEX_TO_NAME = {idx: name for name, idx in POSE_LANDMARKS.items()}

target_index = POSE_LANDMARKS[TARGET_LANDMARK_NAME]

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

typing = False
input_buffer = ""
status_message = ""
status_frames_left = 0

print(f"Controls: '{TYPE_TARGET_KEY}' = type a new target landmark, Enter = confirm, Esc = cancel, '{QUIT_KEY}' = quit")

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
            for idx in POSE_LANDMARKS.values():
                lm = pose_landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), ALL_LANDMARKS_RADIUS, ALL_LANDMARKS_COLOR, -1)

            target_name = INDEX_TO_NAME[target_index]
            target = pose_landmarks[target_index]
            tx, ty = int(target.x * w), int(target.y * h)
            cv2.circle(frame, (tx, ty), TARGET_RADIUS, TARGET_COLOR, TARGET_THICKNESS)
            cv2.putText(
                frame,
                f"{target_name} ({tx}, {ty})",
                (tx + TARGET_RADIUS + 5, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                TARGET_COLOR,
                2,
            )
            print(f"Target pixel: ({tx}, {ty})  norm: ({target.x:.3f}, {target.y:.3f})")

    if typing:
        cv2.putText(
            frame,
            f"New target: {input_buffer}_",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            INPUT_COLOR,
            2,
        )
    elif status_frames_left > 0:
        cv2.putText(
            frame,
            status_message,
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            ERROR_COLOR,
            2,
        )
        status_frames_left -= 1

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if typing:
        if key == 13:  # Enter: confirm typed name
            typed_name = input_buffer.strip().upper()
            if typed_name in POSE_LANDMARKS:
                target_index = POSE_LANDMARKS[typed_name]
                status_message = f"Target set to {typed_name}"
            else:
                status_message = f"Unknown landmark: {typed_name}"
            status_frames_left = STATUS_DISPLAY_FRAMES
            typing = False
            input_buffer = ""
        elif key == 27:  # Esc: cancel typing
            typing = False
            input_buffer = ""
        elif key == 8:  # Backspace
            input_buffer = input_buffer[:-1]
        elif key != 255 and chr(key).isprintable():
            input_buffer += chr(key)
    else:
        if key == ord(QUIT_KEY):
            break
        elif key == ord(TYPE_TARGET_KEY):
            typing = True
            input_buffer = ""

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()