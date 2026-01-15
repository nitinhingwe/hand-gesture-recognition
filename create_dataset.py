import os
import pickle
from collections import Counter
import cv2
import mediapipe as mp

DATA_DIR = "data"
OUT_FILE = "data.pickle"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.2
)

data = []
labels = []
added = Counter()
failed = Counter()

def extract_features(hand_landmarks, handedness_label):
    # Normalize into bbox [0..1]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Avoid division by zero
    dx = (maxx - minx) if (maxx - minx) > 1e-6 else 1e-6
    dy = (maxy - miny) if (maxy - miny) > 1e-6 else 1e-6

    feats = []
    for lm in hand_landmarks.landmark:
        x = (lm.x - minx) / dx
        y = (lm.y - miny) / dy
        feats.append(x)
        feats.append(y)

    # Canonicalize: convert Left to Right by mirroring x
    # (So left & right hands look the same to the model)
    if handedness_label == "Left":
        for i in range(0, len(feats), 2):
            feats[i] = 1.0 - feats[i]

    return feats

for label in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            failed[label] += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            failed[label] += 1
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        handedness_label = "Right"
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label

        feats = extract_features(hand_landmarks, handedness_label)

        if len(feats) == 42:
            data.append(feats)
            labels.append(label)
            added[label] += 1
        else:
            failed[label] += 1

with open(OUT_FILE, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("✅ Saved", OUT_FILE)
print("Added per class:", dict(added))
print("Failed per class:", dict(failed))
print("Final distribution:", dict(Counter(labels)))
