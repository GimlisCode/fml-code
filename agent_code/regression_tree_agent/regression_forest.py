import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from agent_code.regression_tree_agent.regression_tree import RegressionTree
from agent_code.ij_1.data_collector import DataCollector


class RegressionForest:
    def __init__(self, num_of_actions):
        # self.trees = [RegressionTree() for i in range(num_of_actions)]
        # self.trees = [LinearRegression() for i in range(num_of_actions)]
        self.trees = [DecisionTreeRegressor() for i in range(num_of_actions)]

    def train(self, state_t, action, reward, n_min=2):
        for action_idx, tree in enumerate(self.trees):
            # train each tree, using a bootstrap sample of the data
            current_indices = np.argwhere(action == action_idx).squeeze()
            # tree.train(state_t[current_indices], reward[current_indices], n_min)
            tree.fit(state_t[current_indices], reward[current_indices])

    def predict(self, x):
        # compute the ensemble prediction
        predictions = [tree.predict(x.reshape(1, -1)) for tree in self.trees]
        return np.argmax(predictions)


if __name__ == '__main__':
    Q = RegressionForest(6)

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

    state_t[:, 0] = state_t[:, 0] / state_t[:, 0].max()
    state_t[:, 1] = state_t[:, 1] / state_t[:, 1].max()
    state_t[:, 2] = state_t[:, 2] / state_t[:, 2].max()
    state_t[:, 3] = state_t[:, 3] / state_t[:, 3].max()
    state_t[:, 4] = state_t[:, 4] / state_t[:, 4].max()

    Q.train(state_t, action, reward)

    y_pred = np.array([Q.predict(state) for state in state_t])

    print(y_pred)
