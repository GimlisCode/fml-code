import pickle
import random
from typing import Tuple, Optional

import numpy as np
from scipy.spatial.distance import cityblock

from settings import COLS, ROWS


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    try:
        with open("model.pt", "rb") as file:
            self.Q = pickle.load(file)
            print("Loaded")
    except (EOFError, FileNotFoundError):
        self.Q = np.ones((5, 4))


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 40), 0.1)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    action = ACTIONS[np.argmax(self.Q[get_idx_for_state(game_state)])]
    return action


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


def objects_in_bomb_dist(agent_position, obj_positions, dist=3):
    if len(obj_positions) == 0:
        return np.array([])

    steps = get_steps_between(agent_position, obj_positions)
    relevant_distances = steps <= dist

    air_dist = np.abs(obj_positions - agent_position)[relevant_distances]
    steps = steps[relevant_distances]

    dist_x = air_dist[:, 0]
    dist_y = air_dist[:, 1]

    dangerous_bombs = np.logical_or(dist_x == steps, dist_y == steps)

    return obj_positions[relevant_distances][dangerous_bombs]


def get_idx_for_action(action):
    return ACTIONS.index(action)


def get_idx_for_state(game_state: dict):
    our_position = np.array(game_state["self"][3])
    coin_positions = game_state["coins"]

    map = map_game_state_to_image(game_state)

    ret_coins = find_next_coin(map, our_position, coin_positions)

    if ret_coins is None:
        coin_direction_idx = 0
    else:
        coin_direction_idx, _ = ret_coins

    return coin_direction_idx


def map_game_state_to_image(game_state):
    field = (game_state["field"]).copy()  # 0: free tiles, 1: crates, -1: stone walls
    field[field == -1] = 1  # stone walls

    field[game_state["explosion_map"] > 1] = 1  # explosions

    for bomb_pos, bomb_countdown in game_state["bombs"]:
        field[bomb_pos] = 1  # bomb

        if bomb_countdown == 0:
            # explosion in next state
            for idx in objects_in_bomb_dist(bomb_pos, np.argwhere(field != 1)).tolist():
                field[tuple(idx)] = 1
        else:
            # insecure field (but still passable)
            for idx in objects_in_bomb_dist(bomb_pos, np.argwhere(field == 0)).tolist():
                field[tuple(idx)] = 1 + bomb_countdown

    return field  # 0: secure, 1: not passable, 2+: passable, but there will be an explosion in the future


def find_next_coin(map, agent_position, coin_positions) -> Optional[Tuple[int, int]]:
    """
    Returns: next_step_direction, needed_steps
    """

    if not len(coin_positions):
        return None

    steps = get_steps_between(agent_position, coin_positions)

    sorted_indices = np.argsort(steps)

    for idx in sorted_indices:
        is_reachable, next_step_direction, needed_steps = reachable(map, coin_positions[idx], agent_position, limit=30)
        if is_reachable:
            return next_step_direction, needed_steps
    return None


def reachable(map, pos, agent_position, step=0, limit=3) -> Tuple[bool, int, int]:
    sign_x, sign_y = np.sign(pos - agent_position)

    if step > limit:
        return False, 0, 0

    if sign_x == 0 and sign_y == 0:
        return True, 0, step

    if sign_x != 0 and map[agent_position[0] + sign_x, agent_position[1]] != 1:
        ret = reachable(map, pos, np.array([agent_position[0] + sign_x, agent_position[1]]), step+1, limit)
        if ret[0]:
            return True, 1 if sign_x > 0 else 3, ret[2]

    if sign_y != 0 and map[agent_position[0], agent_position[1] + sign_y] != 1:
        ret = reachable(map, pos, np.array([agent_position[0], agent_position[1] + sign_y]), step + 1, limit)
        if ret[0]:
            return True, 4 if sign_y < 0 else 2, ret[2]

    return False, 0, 0
