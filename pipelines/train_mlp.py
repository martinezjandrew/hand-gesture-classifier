"""
train_mlp.py

Builds a multi-layer perceptron model for classifying hand gestures based offof hand landmarks (from mediapipe)

Outputs model to project root directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from HaGridDataset import HaGridDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = HaGridDataset("./annotations/train/")
PATH = "./mlp.pth"
classes = [
    "call",
    "dislike",
    "fist",
    "four",
    "grabbing",
    "grip",
    "hand_heart2",
    "hand_heart",
    "holy",
    "like",
    "little_finger",
    "middle_finger",
    "no_gesture",
    "mute",
    "ok",
    "one",
    "peace",
    "peace_inverted",
    "point",
    "palm",
    "rock",
    "stop",
    "stop_inverted",
    "take_picture",
    "three",
    "three2",
    "three3",
    "three_gun",
    "thumb_index2",
    "thumb_index",
    "timeout",
    "two_up_inverted",
    "two_up",
    "xsign",
]


class HandMLP(nn.Module):
    def __init__(self, input_size=42, num_classes=len(classes)):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    X, y = [], []

    for sample in dataset:
        hands = sample["hand_landmarks"]
        labels = sample["labels"]

        for i, hand in enumerate(hands):
            if hand:
                X.append(np.array(hand).flatten())
                y.append(sample["labels"][i])

    X = torch.tensor(np.stack(X), dtype=torch.float32)

    feature_shape_per_sample = X.shape[1]
    print("Feature shape per sample:", feature_shape_per_sample)

    le = LabelEncoder()
    y_encoded = torch.tensor(le.fit_transform(y), dtype=torch.long)
    train_data = torch.utils.data.TensorDataset(X, y_encoded)
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

    net = HandMLP(input_size=feature_shape_per_sample)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    print("Training complete.")

    torch.save(net.state_dict(), PATH)
