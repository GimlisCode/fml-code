import pickle
import random

import numpy as np
from scipy.spatial.distance import cityblock


from settings import COLS, ROWS


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
        self.Q = np.ones((9, 3, 4, 4, 6)) * 3


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    current_round = game_state["round"]

    near_bomb(np.array(game_state["self"][3]), np.array(game_state["coins"]))

    random_prob = max(.5**(1 + current_round / 15), 0.01)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    return ACTIONS[np.argmax(self.Q[get_idx_for_state(game_state)])]


def get_steps_between(agent_position, object_positions) -> np.ndarray:
    if len(object_positions) == 0:
        return []

    obj_dists_cityblock = np.array([cityblock(agent_position, x) for x in object_positions])
    obj_dist_x_y = object_positions - agent_position

    blocks_vertical = agent_position[0] % 2 == 0
    blocks_horizontal = agent_position[1] % 2 == 0

    same_column = obj_dist_x_y[:, 0] == 0
    same_row = obj_dist_x_y[:, 1] == 0

    obj_dists_cityblock += blocks_vertical * 2 * same_column + blocks_horizontal * 2 * same_row

    same_position = agent_position == object_positions
    obj_dists_cityblock[np.logical_and(same_position[:, 0], same_position[:, 1])] = 0

    return obj_dists_cityblock


def state_to_features(game_state: dict) -> np.array:
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


def near_bomb(agent_position, bomb_positions):
    if len(bomb_positions) == 0:
        return []

    steps = get_steps_between(agent_position, bomb_positions)
    relevant_distances = steps <= 3

    air_dist = np.abs(bomb_positions - agent_position)[relevant_distances]
    steps = steps[relevant_distances]

    dist_x = air_dist[:, 0]
    dist_y = air_dist[:, 1]

    dangerous_bombs = np.logical_or(dist_x == steps, dist_y == steps)

    return bomb_positions[relevant_distances][dangerous_bombs]


def get_idx_for_action(action):
    return ACTIONS.index(action)


def get_idx_for_state(game_state: dict):
    our_position = game_state["self"][3]

    MAX_X = COLS - 2
    MAX_Y = ROWS - 2

    if our_position[0] == 1 and our_position[1] == 1:
        # top left corner
        edge_idx = 0
    elif our_position[0] == MAX_X and our_position[1] == 1:
        # top right corner
        edge_idx = 1
    elif our_position[0] == 1 and our_position[1] == MAX_Y:
        # bottom left corner
        edge_idx = 2
    elif our_position[0] == MAX_X and our_position[1] == MAX_Y:
        # bottom right corner
        edge_idx = 3
    elif our_position[0] == 1:
        # left edge
        edge_idx = 4
    elif our_position[0] == MAX_X:
        # right edge
        edge_idx = 5
    elif our_position[1] == 1:
        # top edge
        edge_idx = 6
    elif our_position[1] == MAX_Y:
        # bottom edge
        edge_idx = 7
    else:
        edge_idx = 8

    if our_position[0] % 2 == 0:
        # vertical blocks
        block_idx = 0
    elif our_position[1] % 2 == 0:
        # horizontal blocks
        block_idx = 1
    else:
        # no blocks
        block_idx = 2

    distances = state_to_features(game_state)

    if distances is None:
        dist_x_idx = 3
        dist_y_idx = 3
    else:
        nearest_coin_dist_x, nearest_coin_dist_y = distances

        if nearest_coin_dist_x > 0:
            dist_x_idx = 0
        elif nearest_coin_dist_x < 0:
            dist_x_idx = 1
        else:
            dist_x_idx = 2

        if nearest_coin_dist_y > 0:
            dist_y_idx = 0
        elif nearest_coin_dist_y < 0:
            dist_y_idx = 1
        else:
            dist_y_idx = 2

    return edge_idx, block_idx, dist_x_idx, dist_y_idx
