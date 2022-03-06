import json

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch


class TrainDataPoint:
    def __init__(self, state_features_t, action, reward, state_features_t_plus_1):
        self.state_features_t = state_features_t
        self.action = action
        self.reward = reward
        self.state_features_t_plus_1 = state_features_t_plus_1

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __hash__(self):
        return hash((tuple(self.state_features_t), self.action, self.reward, tuple(self.state_features_t_plus_1)))


class GameStateDataset(Dataset):
    def __init__(self, state_features_t, actions, rewards, state_features_t_plus_1):
        if (
                len(actions) != len(state_features_t)
                or len(actions) != len(rewards)
                or len(actions) != len(state_features_t_plus_1)
        ):
            raise ValueError("Mismatch in dataset size")

        # self.state_features_t = state_features_t
        # self.actions = actions
        # self.rewards = rewards
        # self.state_features_t_plus_1 = state_features_t_plus_1

        self.data = set()
        for idx in range(len(actions)):
            self.data.add(TrainDataPoint(state_features_t[idx],
                                         actions[idx],
                                         rewards[idx],
                                         state_features_t_plus_1[idx]))

        self.data = list(self.data)

    @staticmethod
    def from_file(file: str) -> 'GameStateDataset':
        with open(file, "r") as f:
            train_data = json.loads(f.read())

        return GameStateDataset(train_data["state_features_t"],
                                train_data["action"],
                                train_data["reward"],
                                train_data["state_features_{t+1}"])

    @staticmethod
    def from_dict(train_data) -> 'GameStateDataset':
        return GameStateDataset(train_data["state_features_t"],
                                train_data["action"],
                                train_data["reward"],
                                train_data["state_features_{t+1}"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T_co:
        feat_t = torch.tensor(self.data[index].state_features_t).double()
        action = self.data[index].action
        reward = self.data[index].reward
        feat_t_plus_1 = torch.tensor(self.data[index].state_features_t_plus_1).double()

        return feat_t, action, reward, feat_t_plus_1

    def to_dict(self):
        data = dict()

        data["state_features_t"] = list()
        data["action"] = list()
        data["reward"] = list()
        data["state_features_{t+1}"] = list()

        for data_point in self.data:
            data["state_features_t"] .append(data_point.state_features_t)
            data["action"].append(data_point.action)
            data["reward"].append(data_point.reward)
            data["state_features_{t+1}"].append(data_point.state_features_t_plus_1)

        return data

