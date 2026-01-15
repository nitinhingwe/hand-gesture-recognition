import os
import cv2
import mediapipe as mp

# -----------------------
# Settings
# -----------------------
DATA_DIR = "data"
CAM_INDEX = 0
WIDTH, HEIGHT = 640, 480
USE_MJPG = True

IMAGES_PER_CLASS = 200  # start with 200 each; later go 400-800 for max accuracy

GESTURES = [
    "open_palm",
    "index_finger",
    "victory",
    "fist",
    "thumbs_up",
]

# -----------------------
# MediaPipe Hands
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# -----------------------
# Setup folders
# -----------------------
os.makedirs(DATA_DIR, exist_ok=True)
for g in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
if USE_MJPG:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("❌ Camera not opened. Try CAM_INDEX=1 or check /dev/video*")

print("✅ Camera started")
print("Controls:")
print("  SPACE = start/stop capture for current class")
print("  N     = next class")
print("  P     = previous class")
print("  Q     = quit")

class_idx = 0
capturing = False
saved_count = {g: len(os.listdir(os.path.join(DATA_DIR, g))) for g in GESTURES}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_name = GESTURES[class_idx]
    target_dir = os.path.join(DATA_DIR, gesture_name)

    hand_ok = results.multi_hand_landmarks is not None

    # Draw landmarks for feedback
    if hand_ok:
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame, lm, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

    # Auto-save only if capturing AND hand detected
    if capturing and hand_ok and saved_count[gesture_name] < IMAGES_PER_CLASS:
        img_name = f"{saved_count[gesture_name]:04d}.jpg"
        cv2.imwrite(os.path.join(target_dir, img_name), frame)
        saved_count[gesture_name] += 1

    status = "CAPTURING ✅" if capturing else "PAUSED ⏸"
    hand_status = "HAND OK ✅" if hand_ok else "NO HAND ❌"

    cv2.putText(frame, f"Class: {gesture_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"{status} | {hand_status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Saved: {saved_count[gesture_name]}/{IMAGES_PER_CLASS}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        capturing = not capturing
    elif key == ord("n"):
        capturing = False
        class_idx = (class_idx + 1) % len(GESTURES)
    elif key == ord("p"):
        capturing = False
        class_idx = (class_idx - 1) % len(GESTURES)

cap.release()
cv2.destroyAllWindows()
print("✅ Done. Saved counts:", saved_count)
