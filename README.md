# Hand Gesture Classifier

Created by: Andrew Martinez

## Description

This **Hand Gesture Classifier** is a PyTorch model trained to classify a hand gesture of an image based off the detected hand landmarks within it.

This model is intended to be a part of another project, which will utilize this model to detect hand gestures presented to a device's screen.

## Requirements

Install the annotations folder from the HaGridDataset into the root of this project, after cloning.

Create virtual environment and pip install everything in `requirements.txt`

## File Structure

- `./HaGridDataset.py` - defines the HaGrid data we will use to train models.
- `./hand_landmarker_demo.py` - simple demo to view how mediapipe views hand landmarks.
- `./start_demo.sh` - linux command to get OpenCV to work with my Arch Linux setup.

- `./pipelines/`
  - contains scripts related to training and testing models

  - _pipelines_
    - `test_mlp.py/` - tests the model created in `train_mlp.py/`
    - `train_mlp.py/` - trains a MLP model on HaGridDataset
    - `train_fr.py/` - trains and test a Random Forest Classifier

- `./show_hand_skeleton.py` - creates a plot graph to present hand landmarks and label(s)
