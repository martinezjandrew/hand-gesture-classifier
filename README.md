# Hand Gesture Classifier

Created by: Andrew Martinez

## Description

This **Hand Gesture Classifier** is a PyTorch model trained to classify a hand gesture of an image based off the detected hand landmarks within it.

This model is intended to be a part of another project, which will utilize this model to detect hand gestures presented to a device's screen.

## Current Status

`./HaGridDataset.py` - defines the HaGrid data we will use to train models.
`./hand_landmarker_demo.py` - simple demo to view how mediapipe views hand landmarks.
`./start_demo.sh` - linux command to get OpenCV to work with my Arch Linux setup.
`./test.py` - quick script to view the handlandmarks from HaGrid in a graph.
`./train.py` - basic script to train a Random Forest Classifier on the HaGridDataset.
