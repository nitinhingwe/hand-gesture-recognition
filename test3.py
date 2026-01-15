import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PICKLE_PATH = "data.pickle"
MODEL_OUT = "gesture_model.joblib"
ENCODER_OUT = "label_encoder.joblib"

with open(DATA_PICKLE_PATH, "rb") as f:
    d = pickle.load(f)

X = np.array(d["data"], dtype=np.float32)
y = np.array(d["labels"])

print("✅ Loaded dataset")
print("X shape:", X.shape)
print("Label distribution:", Counter(y))

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Test Accuracy:", round(acc * 100, 2), "%")
print("\nReport:\n", classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, MODEL_OUT)
joblib.dump(le, ENCODER_OUT)

print("\n✅ Saved model:", MODEL_OUT)
print("✅ Saved encoder:", ENCODER_OUT)
