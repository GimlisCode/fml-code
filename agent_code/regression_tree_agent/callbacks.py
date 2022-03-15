import numpy as np

from .regression_forest import RegressionForest
from agent_code.ij_1.data_collector import DataCollector
from agent_code.ij_1.callbacks import get_idx_for_state


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    self.Q = RegressionForest(6)

    data_loader = DataCollector(filename="combined_data.json")

    state_t = list()
    action = list()
    reward = list()
    for data_point in data_loader.data:
        state_t.append(list(data_point.state_features_t))
        action.append(data_point.action)
        reward.append(data_point.reward)

    state_t = np.squeeze(np.swapaxes(np.array(state_t), 0, 1)).T
    action = np.array(action)
    reward = np.array(reward)

    self.Q.train(state_t, action, reward)


def act(self, game_state: dict) -> str:
    return ACTIONS[self.Q.predict(np.array(get_idx_for_state(game_state)))]
