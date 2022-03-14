from pathlib import Path
import json


class DataCollector:
    def __init__(self, train_data_path: str = "."):
        self.train_data_path = Path(train_data_path)

        try:
            with open(self.train_data_path.joinpath("data.json"), "r") as file:
                self.data = set([TrainDataPoint.from_dict(x) for x in json.loads(file.read())["data"]])
        except FileNotFoundError:
            self.data = set()

    def add(self, state_features_t, action, reward, state_features_t_plus_1):
        self.data.add(TrainDataPoint(state_features_t, action, reward, state_features_t_plus_1))

    def save(self):
        with open(self.train_data_path.joinpath("data.json"), "w") as file:
            file.write(json.dumps({"data": [x.as_dict() for x in self.data]}))


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