import json
import pickle
import random
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cityblock


from settings import COLS, ROWS


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    try:
        with open("model.pt", "rb") as file:
            self.Q = pickle.load(file)
            print("Loaded")
    except (EOFError, FileNotFoundError):
        # self.Q = np.random.rand(9, 3, 3, 3, 4) * 3
        self.Q = np.ones((11, 3, 3, 4)) * 3

    train_data_path = Path("train_data")

    if not train_data_path.exists():
        train_data_path.mkdir()

    idx_so_far = [int(f.name.split("_")[0]) for f in train_data_path.glob("*.tif")]

    self.img_idx = 0 if len(idx_so_far) == 0 else max(idx_so_far) + 1

    try:
        with open("train_data/meta_info.json", "r") as file:
            self.meta_info = [TrainDataPoint.from_dict(x) for x in json.loads(file.read())["meta_info"]]
    except FileNotFoundError:
        self.meta_info = list()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 100), 0.1)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(MOVE_ACTIONS)

    self.logger.debug("Querying model for action.")
    return MOVE_ACTIONS[np.argmax(self.Q[get_idx_for_state(game_state)])]


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


def get_nearest_coin_dist(game_state: dict):
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return [[np.nan, np.nan]]

    our_position = np.array(game_state["self"][3])
    coin_positions = np.array(game_state["coins"])

    if len(coin_positions) == 0:
        return [[np.nan, np.nan]]

    # coin_dists = [cityblock(our_position, x) for x in coin_positions]
    coin_dists = get_steps_between(our_position, coin_positions)

    nearest_coin_dist_x, nearest_coin_dist_y = coin_positions[np.argmin(coin_dists)] - our_position

    return [[nearest_coin_dist_x, nearest_coin_dist_y]]


def get_idx_for_action(action):
    return MOVE_ACTIONS.index(action)


def get_idx_for_state(game_state: dict):
    our_position = game_state["self"][3]

    MAX_X = COLS - 2
    MAX_Y = ROWS - 2

    if our_position[0] == 1 and our_position[1] == 1:
        # top left corner
        pos_idx = 0
    elif our_position[0] == MAX_X and our_position[1] == 1:
        # top right corner
        pos_idx = 1
    elif our_position[0] == 1 and our_position[1] == MAX_Y:
        # bottom left corner
        pos_idx = 2
    elif our_position[0] == MAX_X and our_position[1] == MAX_Y:
        # bottom right corner
        pos_idx = 3
    elif our_position[0] % 2 == 0:
        # vertical blocks
        pos_idx = 4
    elif our_position[1] % 2 == 0:
        # horizontal blocks
        pos_idx = 5
    elif our_position[0] == 1:
        # left edge
        pos_idx = 6
    elif our_position[0] == MAX_X:
        # right edge
        pos_idx = 7
    elif our_position[1] == 1:
        # top edge
        pos_idx = 8
    elif our_position[1] == MAX_Y:
        # bottom edge
        pos_idx = 9
    else:
        # no blocks in our possible moving directions
        pos_idx = 10

    dist_x_idx, dist_y_idx = get_distance_indices(get_nearest_coin_dist(game_state))[0]

    return pos_idx, int(dist_x_idx), int(dist_y_idx)


def get_distance_indices(distances):
    if isinstance(distances, list):
        distances = np.array(distances)

    result = np.ones_like(distances, dtype=int) + 1  # 2 as the else case, i.e. zero distance

    result[np.isnan(distances)] = -1  # i.e. last index

    result[:, 0][distances[:, 0] > 0] = 0
    result[:, 0][distances[:, 0] < 0] = 1

    result[:, 1][distances[:, 1] > 0] = 0
    result[:, 1][distances[:, 1] < 0] = 1

    return result


def get_features_for_state(game_state: dict):
    our_position = game_state["self"][3]

    dist_x_idx, dist_y_idx = get_distance_indices(get_nearest_coin_dist(game_state))[0]

    return *our_position, int(dist_x_idx), int(dist_y_idx)


class TrainDataPoint:
    def __init__(self, state_features_t, action, reward, state_features_t_plus_1):
        self.state_features_t = state_features_t
        self.action = action
        self.reward = reward
        self.state_features_t_plus_1 = state_features_t_plus_1

    @staticmethod
    def from_dict(obj: dict):
        return TrainDataPoint(obj["state_features_t"], obj["action"], obj["reward"], obj["state_features_t_plus_1"])

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
