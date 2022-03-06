import random
import json

import numpy as np
import torch
from scipy.spatial.distance import cityblock


from agent_code.cc_2.network import QNetwork


from settings import COLS, ROWS


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    self.Q = QNetwork(features_in=7, features_out=4)

    #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.device = torch.device('cpu')
    self.Q.to(self.device)

    try:
        with open("train_info.json", "r") as file:
            self.train_info = json.loads(file.read())
    except (EOFError, FileNotFoundError):
        self.train_info = dict()
        self.train_info["train_loss"] = list()

    try:
        self.Q.load("model.pt")
    except:
        pass

    self.Q.eval()


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 15), 0.01)
    if self.train and random.random() < random_prob:
        return np.random.choice(MOVE_ACTIONS)

    features = state_to_features(game_state).to(self.device)
    return MOVE_ACTIONS[torch.argmax(self.Q.forward(features)).cpu()]


def get_steps_between(agent_position, object_positions):
    obj_dists_cityblock = [cityblock(agent_position, x) for x in object_positions]
    obj_dist_x_y = object_positions - agent_position

    blocks_vertical = agent_position[0] % 2 == 0
    blocks_horizontal = agent_position[1] % 2 == 0

    same_column = obj_dist_x_y[:, 0] == 0
    same_row = obj_dist_x_y[:, 1] == 0

    obj_dists_cityblock += blocks_vertical * 2 * same_column + blocks_horizontal * 2 * same_row

    same_position = agent_position == object_positions
    obj_dists_cityblock[np.logical_and(same_position[:, 0], same_position[:, 1])] = 0

    return obj_dists_cityblock


def get_nearest_coin_dist(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    our_position = np.array(game_state["self"][3])
    coin_positions = np.array(game_state["coins"])

    if len(coin_positions) == 0:
        return None

    # coin_dists = [cityblock(our_position, x) for x in coin_positions]
    coin_dists = get_steps_between(our_position, coin_positions)

    nearest_coin_dist_x, nearest_coin_dist_y = coin_positions[np.argmin(coin_dists)] - our_position

    return nearest_coin_dist_x, nearest_coin_dist_y


def get_idx_for_action(action):
    return MOVE_ACTIONS.index(action)


def state_to_features(game_state: dict):
    features = list()

    our_position = game_state["self"][3]

    MAX_X = COLS - 2
    MAX_Y = ROWS - 2

    # relative player position
    features.append(our_position[0] / MAX_X)
    features.append(our_position[1] / MAX_Y)

    features.append(1 if our_position[0] % 2 == 0 else 0)  # vertical blocks
    features.append(1 if our_position[1] % 2 == 0 else 0)  # horizontal blocks

    distances = get_nearest_coin_dist(game_state)

    if distances is None:
        features.append(1)
        features.append(1)
        features.append(0)  # bool that there is no coin
    else:
        features.append(distances[0] / MAX_X)  # relative distance along x-axis
        features.append(distances[1] / MAX_Y)  # relative distance along y-axis
        features.append(1)  # bool that there is no coin

    return torch.tensor(features)  # 7 features
