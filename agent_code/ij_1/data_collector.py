from typing import List
from pathlib import Path
import json


class DataCollector:
    def __init__(self, folder: str = ".", filename: str = "data.json"):
        self.train_data_path = Path(folder)

        if not self.train_data_path.exists():
            self.train_data_path.mkdir()

        self.filename = filename

        try:
            with open(self.train_data_path.joinpath(filename), "r") as file:
                self.data = set([TrainDataPoint.from_dict(x) for x in json.loads(file.read())["data"]])
        except FileNotFoundError:
            self.data = set()

    def add(self, state_features_t, action, reward, state_features_t_plus_1):
        self.data.add(TrainDataPoint(state_features_t, action, reward, state_features_t_plus_1))

    def save(self):
        with open(self.train_data_path.joinpath(self.filename), "w") as file:
            file.write(json.dumps({"data": [x.as_dict() for x in self.data]}))

    @staticmethod
    def combine(files: List[str], save_to: str):
        data = set()
        for collector in [DataCollector(str(Path(file).parent.absolute()), Path(file).name) for file in files]:
            data = data.union(collector.data)

        with open(save_to, "w") as file:
            file.write(json.dumps({"data": [x.as_dict() for x in data]}))


class TrainDataPoint:
    def __init__(self, state_features_t, action, reward, state_features_t_plus_1):
        self.state_features_t = state_features_t
        self.action = action
        self.reward = reward
        self.state_features_t_plus_1 = state_features_t_plus_1

    @staticmethod
    def from_dict(obj: dict):
        return TrainDataPoint(
            tuple(obj["state_features_t"]),
            obj["action"],
            obj["reward"],
            tuple(obj["state_features_t_plus_1"])
        )

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __hash__(self):
        return hash((tuple(self.state_features_t), self.action, self.reward, tuple(self.state_features_t_plus_1)))

    def as_dict(self):
        return {
            "state_features_t": self.state_features_t,
            "action": self.action,
            "reward": self.reward,
            "state_features_t_plus_1": self.state_features_t_plus_1
        }