from sklearn.base import np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from HaGridDataset import HaGridDataset

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
