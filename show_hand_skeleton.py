import numpy as np
from HaGridDataset import HaGridDataset
import matplotlib.pyplot as plt


# Mediapipe-style hand connections
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
]


def show_hand_skeleton(data):
    """
    all_hands_landmarks: list of hands, each hand is list of [[x, y], ...] points
    """
    plt.figure(figsize=(6, 6))

    all_hands_landmarks = data.get("hand_landmarks")

    for hand_idx, hand_landmarks in enumerate(all_hands_landmarks):
        landmarks = np.array(hand_landmarks)

        # Draw points
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=30, c="r")

        # Draw skeleton connections
        for start, end in HAND_CONNECTIONS:
            x = [landmarks[start, 0], landmarks[end, 0]]
            y = [landmarks[start, 1], landmarks[end, 1]]
            plt.plot(x, y, c="b", linewidth=2)

    plt.gca().invert_yaxis()  # flip y-axis to match image coordinates
    plt.axis("equal")
    label = data.get("labels", "Missing")
    plt.title(label, fontsize=14)
    plt.show()


if __name__ == "__main__":
    # Load dataset
    data = HaGridDataset(data_path="./annotations/test/")
    print(f"Dataset size: {len(data)}")

    test_data = data[10000]
    print(test_data)
    # Show all hands in the first sample
    show_hand_skeleton(test_data)
