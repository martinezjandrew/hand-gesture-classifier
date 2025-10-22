"""
test_mlp.py

Loads the mlp model, picked a random index, and evaluations the real vs predicted labels.

Also prints out F1 score, confusion_matrix, and FPR per class.

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from HaGridDataset import HaGridDataset
from sklearn.preprocessing import LabelEncoder
from pipelines.train_mlp import HandMLP, PATH, classes
import random
from test import show_hand_skeleton
from sklearn.metrics import f1_score, confusion_matrix


def main():
    # Load dataset
    dataset = HaGridDataset("./annotations/test/")

    X, y = [], []

    for sample in dataset:
        for hand, label in zip(sample["hand_landmarks"], sample["labels"]):
            if hand:  # skip empty hands
                X.append(np.array(hand).flatten())
                y.append(label)

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    print("Feature shape per sample:", X.shape[1])

    # Encode labels
    le = LabelEncoder()
    y_encoded = torch.tensor(le.fit_transform(y), dtype=torch.long)

    # Create DataLoader
    test_data = TensorDataset(X, y_encoded)
    testloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    # Load model
    net = HandMLP()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    #
    # get random sample and test
    idx = random.randint(0, len(X) - 1)
    landmark = X[idx].unsqueeze(0)  # add batch dimension
    true_label = y[idx]

    with torch.no_grad():
        output = net(landmark)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = classes[predicted_idx]

    print(f"Random sample index: {idx}")
    print(f"True label:      {true_label}")
    print(f"Predicted label: {predicted_label}")
    # visualize hand
    sample = dataset[idx]
    show_hand_skeleton(sample)

    # Compute predictions
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_X, batch_y in testloader:
            outputs = net(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(batch_y.tolist())

    f1 = f1_score(all_true, all_preds, average="macro")
    print(f"F1 score: {f1:.4f}")

    cm = confusion_matrix(all_true, all_preds)
    print("confusion_matrix:\n", cm)

    num_classes = cm.shape[0]
    fpr_list = []
    for i in range(num_classes):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fpr_list.append(fpr)
    print("\nFalse Positive Rate per class:")
    for cls_idx, rate in enumerate(fpr_list):
        print(f"  {classes[cls_idx]}: {rate:.4f}")


if __name__ == "__main__":
    main()
