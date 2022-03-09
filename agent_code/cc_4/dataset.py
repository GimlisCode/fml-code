from pathlib import Path

import tifffile
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch


class GameStateDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

        self.data = list(self.data_path.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T_co:
        filename = self.data[index]

        img = torch.tensor(tifffile.imread(filename))

        state_t = torch.zeros((1, 18, 18), dtype=torch.double)
        state_t_1 = torch.zeros((1, 18, 18), dtype=torch.double)

        state_t = state_t / 4
        state_t_1 = state_t_1 / 4

        state_t[0, 0:-1, 0:-1] = img[0]
        state_t_1[0, 0:-1, 0:-1] = img[1]

        _, action, reward = filename.name.replace(".tif", "").split("_")

        reward = int(reward) / 10  # rewards between -1 and 1

        return state_t, int(action), reward, state_t_1