import cv2
import torch
import numpy as np
from pipelines.train_mlp import HandMLP
import mediapipe as mp

# Load MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HandMLP()
net.load_state_dict(
    torch.load("model_weights.pth", map_location=device, weights_only=False)
)
net.eval()

metadata = torch.load("metadata.pth", map_location=device, weights_only=False)
classes = metadata["classes"]
mean = torch.tensor(metadata["mean"]).float().to(device)
std = torch.tensor(metadata["std"]).float().to(device)

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# OpenCV camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to array
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            x_input = (
                torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            )

            # Predict gesture
            with torch.no_grad():
                output = net(x_input)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_label = classes[pred_idx]

            # Draw landmarks + label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(
                frame,
                pred_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
