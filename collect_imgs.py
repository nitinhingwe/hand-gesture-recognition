import cv2
import os
import mediapipe as mp

CAM_INDEX = 0
DATA_DIR = "data"
IMAGES_PER_CLASS = 200

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press C = capture (only if hand detected), Q = quit")


while True:
    label = input("\nEnter gesture name (open_palm, index_finger, victory, fist, thumbs_up) or q: ")
    if label == "q":
        break

    save_dir = os.path.join(DATA_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_ok = results.multi_hand_landmarks is not None

        status = "HAND OK ?" if hand_ok else "NO HAND ?"
        cv2.putText(frame, f"{label} {count}/{IMAGES_PER_CLASS}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Collect Gesture Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if hand_ok:
                cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), frame)
                count += 1
            else:
                print("Skipped capture: hand not detected")
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
