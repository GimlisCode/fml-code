import pickle
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent_code.ij_1.data_collector import DataCollector


class Trainer:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.5, epochs: int = 2, save_snapshots: bool = False):
        self.alpha = alpha
        self.gamma = gamma
        self.epochs = epochs
        self.save_snapshots = save_snapshots

        self.Q = np.zeros((6, 2, 6, 5, 6, 6), dtype=float)

        self.data_loader = DataCollector()

    def update_Q(self, idx_t, action_idx, reward, idx_t1):
        loss = reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx]
        self.Q[idx_t][action_idx] += self.alpha * loss
        return abs(loss)

    def train(self):
        data = list(self.data_loader.data)

        print(f"Number of data samples: {len(data)}")

        averaged_loss = list()
        loss = list()

        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0

            for data_point in data:
                current_loss = self.update_Q(
                    data_point.state_features_t,
                    data_point.action,
                    data_point.reward,
                    data_point.state_features_t_plus_1
                )
                epoch_loss += current_loss
                loss.append(current_loss)

            shuffle(data)

            averaged_loss.append(epoch_loss / len(data))

            if self.save_snapshots:
                self.save_model(filename=f"snapshot_{epoch}.pt")

        plt.plot(averaged_loss, label="training error")
        plt.xlabel("epochs")
        plt.ylabel("error")
        # plt.legend()
        plt.savefig("replaying_loss.pdf", bbox_inches="tight")
        plt.show()

    def save_model(self, filename: str = "model_train_data.pt"):
        with open(filename, "wb") as file:
            pickle.dump(self.Q, file)


if __name__ == '__main__':
    DataCollector.combine(files=[
        "../../ExperimentData/environments/trainedAbove/train_data/train_data0.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data1.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data2.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data3.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data4.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data5.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data6.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data7.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data8.json",
        "../../ExperimentData/environments/trainedAbove/train_data/train_data9.json",
    ], save_to="data.json")

    trainer = Trainer(epochs=35, save_snapshots=False, alpha=0.1)
    trainer.train()
    trainer.save_model()
