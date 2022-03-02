import os
import pickle
import random

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

    self.last_nearest_coin_dist = -1

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = .25
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(MOVE_ACTIONS)

    self.logger.debug("Querying model for action.")
    return MOVE_ACTIONS[np.argmax(self.Q[get_idx_for_state(game_state)])]


def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    our_position = np.array(game_state["self"][3])
    coin_positions = np.array(game_state["coins"])

    coin_dists = [cityblock(our_position, x) for x in coin_positions]
    # nearest_coin_dist = np.min(coin_dists)
    nearest_coin_dist_x, nearest_coin_dist_y = coin_positions[np.argmin(coin_dists)] - our_position

    return nearest_coin_dist_x, nearest_coin_dist_y


def get_idx_for_action(action):
    return MOVE_ACTIONS.index(action)


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

    nearest_coin_dist_x, nearest_coin_dist_y = state_to_features(game_state)

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
