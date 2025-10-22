from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import json


class HaGridDataset(Dataset):
    """HaGrid dataset."""

    def __init__(self, data_path):
        data = []

        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)

            if file.startswith(".") or not os.path.isfile(file_path):
                continue

            with open(os.path.join(data_path, file), "r") as f:
                gesture_data = json.load(f)
                data.extend(gesture_data.values())

        self.frame = pd.DataFrame(data)[["labels", "united_label", "hand_landmarks"]]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        item = self.frame.iloc[index]

        return item.to_dict()
