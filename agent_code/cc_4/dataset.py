from pathlib import Path
from random import shuffle
from typing import Tuple

import tifffile
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch


class GameStateDataset(Dataset):
    def __init__(self, files: list, device: torch.device = None, load=True):
        self.files = files
        self.device = device
        shuffle(self.files)

        self.state_t = list()
        self.state_t_1 = list()
        self.action = list()
        self.reward = list()

        if load:
            self.load_files()
            self._move_data_to(device)
            self.loaded = True
        else:
            self.loaded = False

    @staticmethod
    def from_data_path(data_path: str, device: torch.device = None, load: bool = True) -> "GameStateDataset":
        return GameStateDataset(list(Path(data_path).glob("*.tif")), device=device, load=load)

    def load_files(self):
        for filename in self.files:
            img = torch.tensor(tifffile.imread(filename))

            state_t = torch.zeros((4, 18, 18), dtype=torch.double)
            state_t_1 = torch.zeros((4, 18, 18), dtype=torch.double)

            state_t[:, 0:-1, 0:-1] = img[:4]
            state_t_1[:, 0:-1, 0:-1] = img[4:]

            _, action, reward = filename.name.replace(".tif", "").split("_")

            action = torch.tensor(int(action))
            reward = torch.tensor(int(reward) / 10)  # rewards between -1 and 1

            self.state_t.append(state_t)
            self.action.append(action)
            self.reward.append(reward)
            self.state_t_1.append(state_t_1)

    def _move_data_to(self, device):
        if device is not None:
            for idx in range(len(self.state_t)):
                self.state_t[idx] = self.state_t[idx].to(device)
                self.state_t_1[idx] = self.state_t_1[idx].to(device)
                self.action[idx] = self.action[idx].to(device)
                self.reward[idx] = self.reward[idx].to(device)

    def __len__(self) -> int:
        return len(self.state_t)

    def __getitem__(self, index: int) -> T_co:
        if not self.loaded:
            raise ValueError("The data is not loaded. Call load_files() before usage.")
        return self.state_t[index], self.action[index], self.reward[index], self.state_t_1[index]

    def try_to_overcome_bias(self):
        reward_10_data = [f for f in self.files if f.name.endswith("_10.tif")]
        reward_5_data = [f for f in self.files if f.name.endswith("_5.tif")]
        reward_minus_5_data = [f for f in self.files if f.name.endswith("_-5.tif")]
        reward_minus_10_data = [f for f in self.files if f.name.endswith("_-10.tif")]

        min_len = min(len(reward_10_data), len(reward_5_data), len(reward_minus_5_data), len(reward_minus_10_data))

        self.files = (
                reward_10_data[:min_len]
                + reward_5_data[:min_len]
                + reward_minus_5_data[:min_len]
                + reward_minus_10_data[:min_len]
        )

    def split(self, val_percentage: float = 0.2) -> Tuple["GameStateDataset", "GameStateDataset"]:
        split_idx = min(int((1 - val_percentage) * len(self.files)) + 1, len(self.files))
        train_data = self.files[:split_idx]
        val_data = self.files[split_idx:]

        return (
            GameStateDataset(train_data, device=self.device, load=True),
            GameStateDataset(val_data, device=self.device, load=True)
        )
