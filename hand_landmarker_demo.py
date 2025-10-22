import cv2
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Download model if it doesn't exist
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(model_url, model_path)

# Setup the hand landmarker
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path), num_hands=2
)

detector = HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    # Draw landmarks and print handedness
    if result.hand_landmarks:
        for idx, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[idx][0].category_name  # 'Left' or 'Right'
            if handedness == "Left":
                handedness = "Right"
            else:
                handedness = "Left"
            print(f"Detected {handedness} hand")

            h, w, _ = frame.shape
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Label hand on frame
            cv2.putText(
                frame,
                handedness,
                (10, 30 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
