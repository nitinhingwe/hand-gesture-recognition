import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, Counter

# Camera
CAM_INDEX = 0
WIDTH, HEIGHT = 640, 480
USE_MJPG = True

MODEL_PATH = "gesture_model.joblib"
ENCODER_PATH = "label_encoder.joblib"

SMOOTHING_WINDOW = 10
MIN_CONFIDENCE = 0.55

# Friendly display names (folder -> display)
DISPLAY_NAME = {
    "open_palm": "Open Palm",
    "index_finger": "Index Finger",
    "victory": "Victory ✌️",
    "fist": "Closed Fist",
    "thumbs_up": "Thumbs Up 👍"
}

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

print("✅ Loaded model + encoder")
print("Classes:", list(le.classes_))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def extract_features(hand_landmarks, handedness_label):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    dx = (maxx - minx) if (maxx - minx) > 1e-6 else 1e-6
    dy = (maxy - miny) if (maxy - miny) > 1e-6 else 1e-6

    feats = []
    for lm in hand_landmarks.landmark:
        x = (lm.x - minx) / dx
        y = (lm.y - miny) / dy
        feats.append(x)
        feats.append(y)

    if handedness_label == "Left":
        for i in range(0, len(feats), 2):
            feats[i] = 1.0 - feats[i]

    return feats, (minx, miny, maxx, maxy)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
if USE_MJPG:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open camera index {CAM_INDEX}. Try 1 or check /dev/video*")

pred_history = deque(maxlen=SMOOTHING_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_text = "No hand"
    conf_text = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        handedness_label = "Right"
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label

        feats, (minx, miny, maxx, maxy) = extract_features(hand_landmarks, handedness_label)

        if len(feats) == 42:
            X = np.array(feats, dtype=np.float32).reshape(1, -1)

            pred_idx = int(model.predict(X)[0])
            pred_label = le.inverse_transform([pred_idx])[0]

            conf = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                conf = float(np.max(proba))

            if conf is not None and conf < MIN_CONFIDENCE:
                pred_history.append("Unknown")
            else:
                pred_history.append(pred_label)

            vote = Counter(pred_history).most_common(1)[0][0]

            if vote == "Unknown":
                display_text = "Pred: Unknown"
            else:
                display_text = f"Pred: {DISPLAY_NAME.get(vote, vote)}"

            if conf is not None:
                conf_text = f"Conf: {conf*100:.1f}%"

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            x_min = int(minx * w)
            y_min = int(miny * h)
            x_max = int(maxx * w)
            y_max = int(maxy * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    else:
        pred_history.clear()

    cv2.putText(frame, display_text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if conf_text:
        cv2.putText(frame, conf_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition (USB Cam)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
