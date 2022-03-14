import pickle

import numpy as np
from tqdm import tqdm

from agent_code.ij_1.data_collector import DataCollector


class Trainer:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.5, epochs: int = 2):
        self.alpha = alpha
        self.gamma = gamma
        self.epochs = epochs

        self.Q = np.zeros((6, 2, 6, 5, 6, 6))

        self.data_loader = DataCollector()

    def update_Q(self, idx_t, action_idx, reward, idx_t1):
        self.Q[idx_t][action_idx] += self.alpha * (reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx])

    def train(self):
        for _ in tqdm(range(self.epochs)):
            for data_point in self.data_loader.data:
                self.update_Q(
                    data_point.state_features_t,
                    data_point.action,
                    data_point.reward,
                    data_point.state_features_t_plus_1
                )

    def save_model(self, filename: str = "model_train_data.pt"):
        with open(filename, "wb") as file:
            pickle.dump(self.Q, file)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.save_model()
