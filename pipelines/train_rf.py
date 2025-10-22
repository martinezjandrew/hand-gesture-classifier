import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from HaGridDataset import HaGridDataset
from sklearn.metrics import f1_score, confusion_matrix


def main():
    # Load Data

    data = HaGridDataset("./annotations/test/")
    print(f"Loaded data: {len(data)} items")

    # Prepare Data

    x = []
    y = []

    for sample in data:
        hands = sample.get("hand_landmarks")
        labels = sample.get("labels")

        if not hands:
            continue

        for i, hand in enumerate(hands):
            if not hand or len(hand) == 0:
                continue  # skip empty hand data

            hand = np.array(hand).flatten()

            # only keep if shape matches expected (e.g., 42 values = 14 landmarks × 3 coords)
            if hand.shape[0] not in [42, 63]:  # adjust if needed
                print(f"Skipping malformed sample, shape: {hand.shape}")
                continue

            x.append(hand)
            y.append(labels[i])

    # Encode labels as integers

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42
    )

    # Train Classifier

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    # Evaluate

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", accuracy)

    # Predict

    sample_hand = data[0].get("hand_landmarks")[0]
    sample_features = np.array(sample_hand).flatten().reshape(1, -1)
    pred_label = le.inverse_transform(clf.predict(sample_features))
    print("Predicted:", pred_label[0])

    # Evaluate
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", accuracy)

    # F1 score (macro average over all classes)
    f1 = f1_score(y_test, y_pred, average="macro")
    print("F1 score (macro):", f1)

    cm = confusion_matrix(y_test, y_pred)
    num_classes = cm.shape[0]

    fpr_list = []
    for i in range(num_classes):
        FP = cm[:, i].sum() - cm[i, i]  # predicted as i but actually another class
        TN = cm.sum() - (
            cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        )  # all other correct negatives
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fpr_list.append(fpr)

    print("\nFalse Positive Rate per class:")
    for idx, rate in enumerate(fpr_list):
        print(f"  {le.inverse_transform([idx])[0]}: {rate:.4f}")


if __name__ == "__main__":
    main()
