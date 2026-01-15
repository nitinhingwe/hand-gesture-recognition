# hand-gesture-recognition
Hand gesture recognition in RPI5 with USB Camera. Five gestures: Open Palm,  Index Finger,  Victory Sign,  Closed Fist,  &amp; Thumbs Up

# What updates we need in your model files

We will update all 4 parts:

✅ collect_imgs.py

Collect data for 5 gestures (folders)

Only save images when a hand is detected (so no garbage frames)

Show overlay with which gesture you're collecting

✅ create_dataset.py

Extract 42 features (21 landmarks x,y)

Normalize features properly using min/max

Canonicalize handedness (Left mirrored to Right) so both hands work

✅ test3.py

Train a classifier on 5 gestures (RandomForest is perfect on Pi)

Save gesture_model.joblib + label_encoder.joblib

Print class distribution and accuracy

✅ test4.py

Live prediction with smoothing

Show names like Open Palm instead of raw labels

Use same handedness normalization as dataset creation

# Folder structure we will use

Inside your project:

data/
  open_palm/
  index_finger/
  victory/
  fist/
  thumbs_up/


These folder names become the labels automatically (no more 0/1/2 headaches).

# How to proceed step-by-step (exact workflow)
## Step A: Collect data
source .venv/bin/activate
python collect_imgs.py


Capture at least:

200 per gesture minimum

Later upgrade to 500–1000 per gesture for strong accuracy

Tips:

vary distance (near/far)

vary angle (slight tilt)

vary lighting

use both hands sometimes (even though we canonicalize)

## Step B: Create dataset
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python create_dataset.py


Make sure output shows something like:

Added per class: each should be close to your image count
If one class has very low “Added”, it means MediaPipe isn’t detecting it well.

## Step C: Train model
python test3.py

## Step D: Live test
python test4.py

# Mini FAQ
“Should I use left hand or right hand?”

With the handedness mirroring in create_dataset.py + test4.py:
✅ You can use either hand.
Still, adding both hands in dataset makes it even more robust.

“Why mirror at all?”

Because otherwise Left vs Right becomes a different pattern, and the classifier may treat them as different gestures.

### Intergalactic Garage
