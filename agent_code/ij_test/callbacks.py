import pickle

import numpy as np

from agent_code.ij_1.callbacks import get_idx_for_state


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class Direction:
    IS_AT = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4
    UNREACHABLE = 5


class CoinDirection:
    UNREACHABLE_OR_NONE = 0
    RIGHT = Direction.RIGHT
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    UP = Direction.UP


class CrateDirection:
    NEXT_TO = 0
    RIGHT = Direction.RIGHT
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    UP = Direction.UP
    UNREACHABLE_OR_NONE = Direction.UNREACHABLE


class SafeFieldDirection:
    IS_AT = Direction.IS_AT
    RIGHT = Direction.RIGHT
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    UP = Direction.UP
    UNREACHABLE = Direction.UNREACHABLE


class NearestAgentDirection:
    IN_BOMB_RANGE = 0
    RIGHT = Direction.RIGHT
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    UP = Direction.UP
    UNREACHABLE_OR_NONE = Direction.UNREACHABLE


def setup(self):
    self.experiment_number = 0

    with open(f"model{self.experiment_number}.pt", "rb") as file:
        self.Q = pickle.load(file)


def act(self, game_state: dict) -> str:
    return ACTIONS[np.argmax(self.Q[get_idx_for_state(game_state)])]
