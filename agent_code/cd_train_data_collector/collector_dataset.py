import json


class CollectorDataset:
    def __init__(self, data: set = None):
        if data is None:
            self.data = set()
        else:
            self.data: set = data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def from_json(file_path: str) -> "CollectorDataset":
        data = set()

        with open(file_path, "r") as f:
            for element in json.loads(f.read())["data"]:
                data.add(DataElement(element[0], element[1], element[2], element[3]))

        return CollectorDataset(data)

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps({"data": [el.to_list() for el in self.data]}))

    def add(self, features_t, action, reward, features_t1):
        self.data.add(DataElement(features_t, action, reward, features_t1))


class DataElement:
    def __init__(self, features_t, action, reward, features_t1):
        self.features_t = features_t
        self.action = action
        self.reward = reward
        self.features_t1 = features_t1

    def to_list(self):
        return [self.features_t, self.action, self.reward, self.features_t1]

    def __hash__(self):
        return hash((tuple(self.features_t), self.action, self.reward, tuple(self.features_t1)))
