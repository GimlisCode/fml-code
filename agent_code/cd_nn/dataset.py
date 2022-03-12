import json
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class GameStateDataset(Dataset):
    def __init__(self, data: list = None):
        if data is None:
            self.data = list()
        else:
            self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> T_co:
        return self.data[index].get()

    @staticmethod
    def from_json(file_path: str) -> "GameStateDataset":
        data = list()

        with open(file_path, "r") as f:
            for element in json.loads(f.read())["data"]:
                data.append(DataElement(element[0], element[1], element[2], element[3]))

        return GameStateDataset(data)

    def split(self, val_percentage: float = 0.2) -> Tuple["GameStateDataset", "GameStateDataset"]:
        split_idx = min(int((1 - val_percentage) * len(self.data)) + 1, len(self.data))
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]

        return GameStateDataset(train_data), GameStateDataset(val_data)


class DataElement:
    def __init__(self, features_t, action, reward, features_t1):
        self.features_t = torch.tensor(features_t, dtype=torch.double)
        self.action = action
        self.reward = reward / 4
        self.features_t1 = torch.tensor(features_t1, dtype=torch.double)

    def get(self):
        return self.features_t, self.action, self.reward, self.features_t1
