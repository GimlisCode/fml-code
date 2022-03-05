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
        self.Q = np.ones((9, 3, 4, 4, 4, 4, 4, 4, 6)) * 3


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 400), 0.01)
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


def get_k_nearest_object_positions(agent_position, object_positions, k=1) -> list:
    if len(object_positions) == 0:
        return [None] * k

    k_orig = k
    k = min(k, len(object_positions))

    return object_positions[np.argpartition(get_steps_between(agent_position, object_positions), kth=k-1)[:k]].tolist() + [None] * (k_orig - k)


def objects_in_bomb_dist(agent_position, obj_positions, dist=3):
    if len(obj_positions) == 0:
        return []

    steps = get_steps_between(agent_position, obj_positions)
    relevant_distances = steps <= dist

    air_dist = np.abs(obj_positions - agent_position)[relevant_distances]
    steps = steps[relevant_distances]

    dist_x = air_dist[:, 0]
    dist_y = air_dist[:, 1]

    dangerous_bombs = np.logical_or(dist_x == steps, dist_y == steps)

    return obj_positions[relevant_distances][dangerous_bombs]


def get_k_nearest_bombs(agent_position, bomb_positions, k) -> list:
    dangerous_bombs = objects_in_bomb_dist(agent_position, bomb_positions, dist=3)

    if len(dangerous_bombs) == 0:
        return [None] * k

    k_orig = k
    k = min(k, len(dangerous_bombs))

    return dangerous_bombs[np.argpartition(get_steps_between(agent_position, dangerous_bombs), kth=k-1)[:k]].tolist() + [None] * (k_orig - k)


def get_idx_for_action(action):
    return ACTIONS.index(action)


def extract_crate_positions(field):
    return np.argwhere(field == 1)


def get_idx_for_state(game_state: dict):
    # number_of_near_bombs?
    # number_of_near_crates?
    our_position = game_state["self"][3]
    crate_positions = extract_crate_positions(game_state["field"])
    bomb_positions = np.array([coords for coords, _ in game_state["bombs"]])

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

    coin_dist_x_idx, coin_dist_y_idx = get_distance_indices(our_position, get_nearest_coin_dist(game_state))

    nearest_crates = get_k_nearest_object_positions(our_position, crate_positions, k=1)

    crate_indices = list()
    for crate_dist in nearest_crates:
        crate_indices.append(get_distance_indices(our_position, crate_dist))

    nearest_bombs = get_k_nearest_bombs(our_position, bomb_positions, k=1)

    bomb_indices = list()
    for bomb_dist in nearest_bombs:
        bomb_indices.append(get_distance_indices(our_position, bomb_dist))

    return edge_idx, block_idx, coin_dist_x_idx, coin_dist_y_idx, *crate_indices[0], *bomb_indices[0]
    # return edge_idx, block_idx, *crate_indices[0], *bomb_indices[0]


def get_distance_indices(agent_position, object_positions):
    if object_positions is None:
        dist_x_idx = 3
        dist_y_idx = 3
    else:
        dist_x, dist_y = object_positions[0] - agent_position[0], object_positions[1] - agent_position[1]

        if dist_x > 0:
            dist_x_idx = 0
        elif dist_x < 0:
            dist_x_idx = 1
        else:
            dist_x_idx = 2

        if dist_y > 0:
            dist_y_idx = 0
        elif dist_y < 0:
            dist_y_idx = 1
        else:
            dist_y_idx = 2

    return dist_x_idx, dist_y_idx
