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
        # self.Q = np.ones((11, 4, 4, 4, 4, 4, 4, 16, 6)) * 3
        # self.Q = np.ones((11, 4, 4, 4, 4, 16, 6)) * 3
        # self.Q = np.ones((16, 4, 4, 6, 2, 16, 4, 4, 6)) * 3
        self.Q = np.ones((16, 4, 4, 4, 4, 16, 2, 6)) * 3


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 20), 0.01)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        if action == "BOMB" and is_in_corner(game_state["self"][3]):
            action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
        return action

    self.logger.debug("Querying model for action.")
    action = ACTIONS[np.argmax(self.Q[get_idx_for_state(game_state)])]
    if action == "BOMB" and is_in_corner(game_state["self"][3]):
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
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


def get_k_nearest_object_positions(agent_position, object_positions, k=1) -> list:
    if len(object_positions) == 0:
        return [[np.nan, np.nan]] * k

    k_orig = k
    k = min(k, len(object_positions))

    return object_positions[np.argpartition(get_steps_between(agent_position, object_positions), kth=k-1)[:k]].tolist() + [[np.nan, np.nan]] * (k_orig - k)


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
        return [[np.nan, np.nan]] * k

    k_orig = k
    k = min(k, len(dangerous_bombs))

    return dangerous_bombs[np.argpartition(get_steps_between(agent_position, dangerous_bombs), kth=k-1)[:k]].tolist() + [[np.nan, np.nan]] * (k_orig - k)


def get_idx_for_action(action):
    return ACTIONS.index(action)


def extract_crate_positions(field):
    return np.argwhere(field == 1)


class Position:
    TOP_LEFT_CORNER = 0
    TOP_RIGHT_CORNER = 1
    BOTTOM_LEFT_CORNER = 2
    BOTTOM_RIGHT_CORNER = 3
    VERTICAL_BLOCKS = 4
    HORIZONTAL_BLOCKS = 5
    LEFT_EDGE = 6
    RIGHT_EDGE = 7
    TOP_EDGE = 8
    BOTTOM_EDGE = 9
    NO_BLOCKS_AROUND = 10


def is_in_corner(agent_position):
    return agent_position[0] == 1 and agent_position[1] == 1 or agent_position[0] == COLS-2 and agent_position[1] == 1 or agent_position[0] == 1 and agent_position[1] == ROWS-2 or agent_position[0] == COLS-2 and agent_position[1] == ROWS-2


def get_idx_for_state(game_state: dict):
    our_position = np.array(game_state["self"][3])
    crate_positions = extract_crate_positions(game_state["field"])
    bomb_positions = np.array([coords for coords, _ in game_state["bombs"]])

    MAX_X = COLS - 2
    MAX_Y = ROWS - 2

    # if our_position[0] == 1 and our_position[1] == 1:
    #     pos_idx = Position.TOP_LEFT_CORNER
    # elif our_position[0] == MAX_X and our_position[1] == 1:
    #     pos_idx = Position.TOP_RIGHT_CORNER
    # elif our_position[0] == 1 and our_position[1] == MAX_Y:
    #     pos_idx = Position.BOTTOM_LEFT_CORNER
    # elif our_position[0] == MAX_X and our_position[1] == MAX_Y:
    #     pos_idx = Position.BOTTOM_RIGHT_CORNER
    # elif our_position[0] % 2 == 0:
    #     pos_idx = Position.VERTICAL_BLOCKS
    # elif our_position[1] % 2 == 0:
    #     pos_idx = Position.HORIZONTAL_BLOCKS
    # elif our_position[0] == 1:
    #     pos_idx = Position.LEFT_EDGE
    # elif our_position[0] == MAX_X:
    #     pos_idx = Position.RIGHT_EDGE
    # elif our_position[1] == 1:
    #     pos_idx = Position.TOP_EDGE
    # elif our_position[1] == MAX_Y:
    #     pos_idx = Position.BOTTOM_EDGE
    # else:
    #     # no blocks in our possible moving directions
    #     pos_idx = Position.NO_BLOCKS_AROUND

    # coin_dist_x_idx, coin_dist_y_idx = get_distance_indices(get_nearest_coin_dist(game_state))[0]

    nearest_crates = get_k_nearest_object_positions(our_position, crate_positions, k=1)
    crate_indices = get_distance_indices(nearest_crates - our_position)

    nearest_bombs = get_k_nearest_bombs(our_position, bomb_positions, k=1)
    bomb_indices = get_distance_indices(nearest_bombs - our_position)

    expl_idx = get_explosion_indices(our_position, game_state["explosion_map"])

    in_bomb_range = 1 if is_in_bomb_range(our_position, bomb_positions) else 0
    # neighbor_in_bomb_range_idx = get_dangerous_neighbor_indices(our_position, bomb_positions)

    # neighbor_in_danger_index = get_dangerous_neighbor_index(game_state)

    # direction_with_most_crates = find_direction_to_increase_crates_destroyed(game_state)
    movement_idx = get_movement_indices(game_state)

    # return pos_idx, coin_dist_x_idx, coin_dist_y_idx, *crate_indices[0], *bomb_indices[0], expl_idx
    # return movement_idx, *crate_indices[0], direction_with_most_crates, in_bomb_range, neighbor_in_danger_index, *bomb_indices[0]
    # return movement_idx, *crate_indices[0], in_bomb_range, neighbor_in_danger_index, *bomb_indices[0]
    return movement_idx, *crate_indices[0], *bomb_indices[0], expl_idx, in_bomb_range


def is_next_to_crate(agent_position, crate_positions):
    col = agent_position[0]
    row = agent_position[1]

    top_pos = (col, row - 1)
    right_pos = (col + 1, row)
    bottom_pos = (col, row + 1)
    left_pos = (col - 1, row)

    return top_pos in crate_positions.tolist() or right_pos in crate_positions.tolist() or bottom_pos in crate_positions.tolist() or left_pos in crate_positions.tolist()


def is_in_bomb_range(agent_position, bomb_positions):
    return len(bomb_positions) > 0 and len(objects_in_bomb_dist(agent_position, bomb_positions)) > 0


def find_direction_to_increase_crates_destroyed(game_state):
    agent_position = game_state["self"][3]
    crate_positions = extract_crate_positions(game_state["field"])
    crate_positions_list = crate_positions.tolist()

    col = agent_position[0]
    row = agent_position[1]
    top_idx = 0
    right_idx = 1
    bottom_idx = 2
    left_idx = 3
    current_score = 0
    scores = np.array([0, 0, 0, 0])

    top_pos = (col, row - 1)
    right_pos = (col + 1, row)
    bottom_pos = (col, row + 1)
    left_pos = (col - 1, row)

    if not has_vertical_blocks(agent_position):
        if row != 1 and top_pos not in crate_positions_list:
            # go top
            scores[top_idx] = len(objects_in_bomb_dist(top_pos, crate_positions))
        if row != ROWS-2 and bottom_pos not in crate_positions_list:
            # go bottom
            scores[bottom_idx] = len(objects_in_bomb_dist((col, row+1), crate_positions))
    if not has_horizontal_blocks(agent_position):
        if col != 1 and left_pos not in crate_positions_list:
            # go left
            scores[left_idx] = len(objects_in_bomb_dist((col-1, row), crate_positions))
        if col != COLS-2 and right_pos not in crate_positions_list:
            # go right
            scores[right_idx] = len(objects_in_bomb_dist((col+1, row), crate_positions))

    current_score = len(objects_in_bomb_dist(agent_position, crate_positions))
    if np.amax(scores) > current_score:
        return np.argmax(scores)
    return 5


def has_vertical_blocks(agent_position):
    return agent_position[0] % 2 == 0


def has_horizontal_blocks(agent_position):
    return agent_position[1] % 2 == 0

# def are_objects_in_sight(agent_position, object_positions):
#     obj_dist_x_y = object_positions - agent_position
#
#     blocks_not_vertical = agent_position[0] % 2 != 0
#     blocks_not_horizontal = agent_position[1] % 2 != 0
#
#     same_column = obj_dist_x_y[:, 0] == 0
#     same_row = obj_dist_x_y[:, 1] == 0
#
#     return np.logical_or(np.logical_and(same_column, blocks_not_vertical),
#                          np.logical_and(same_row, blocks_not_horizontal))


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


def get_explosion_indices(agent_position, explosion_map):
    x, y = agent_position

    left = 1 if explosion_map[x - 1, y] != 0 else 0
    right = 2 if explosion_map[x + 1, y] != 0 else 0
    top = 4 if explosion_map[x, y - 1] != 0 else 0
    bottom = 8 if explosion_map[x, y + 1] != 0 else 0

    return left & right & top & bottom


def get_dangerous_neighbor_index(game_state):
    agent_position = game_state["self"][3]
    explosion_map = game_state["explosion_map"]
    bomb_positions = np.array([coordinates for coordinates, _ in game_state["bombs"]])

    col = agent_position[0]
    row = agent_position[1]
    # top_left_pos = (col-1, row-1)
    top_pos = (col, row - 1)
    # top_right_pos = (col+1, row-1)
    right_pos = (col + 1, row)
    # bottom_left_pos = (col-1, row+1)
    bottom_pos = (col, row + 1)
    # bottom_right_pos = (col+1, row+1)
    left_pos = (col - 1, row)

    top = 1 if explosion_map[top_pos] or is_in_bomb_range(top_pos, bomb_positions) else 0
    bottom = 2 if explosion_map[bottom_pos] or is_in_bomb_range(bottom_pos, bomb_positions) else 0
    left = 4 if explosion_map[left_pos] or is_in_bomb_range(left_pos, bomb_positions) else 0
    right = 8 if explosion_map[right_pos] or is_in_bomb_range(right_pos, bomb_positions) else 0
    # top_left_pos = 16 if explosion_map[top_left_pos] or is_in_bomb_range(top_left_pos, bomb_positions) else 0
    # top_right_pos = 24 if explosion_map[top_right_pos] or is_in_bomb_range(top_right_pos, bomb_positions) else 0
    # bottom_left_pos = 56 if explosion_map[bottom_left_pos] or is_in_bomb_range(bottom_left_pos, bomb_positions) else 0
    # bottom_right_pos = 128 if explosion_map[bottom_right_pos] or is_in_bomb_range(bottom_right_pos, bomb_positions) else 0

    return left & right & top & bottom # & top_left_pos & top_right_pos & bottom_left_pos & bottom_right_pos


def get_movement_indices(game_state):
    agent_position = game_state["self"][3]
    crate_positions = extract_crate_positions(game_state["field"]).tolist()

    col = agent_position[0]
    row = agent_position[1]
    top_pos = (col, row - 1)
    right_pos = (col + 1, row)
    bottom_pos = (col, row + 1)
    left_pos = (col - 1, row)

    top = 0
    right = 0
    bottom = 0
    left = 0

    if not has_vertical_blocks(agent_position):
        if row != 1 and top_pos not in crate_positions:
            # go top
            top = 1
        if row != ROWS - 2 and bottom_pos not in crate_positions:
            # go bottom
            bottom = 2
    if not has_horizontal_blocks(agent_position):
        if col != 1 and left_pos not in crate_positions:
            # go left
            left = 4
        if col != COLS - 2 and right_pos not in crate_positions:
            # go right
            right = 8
    return left & right & top & bottom

