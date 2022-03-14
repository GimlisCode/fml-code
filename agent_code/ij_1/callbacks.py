import pickle
import random
from typing import Tuple, Optional

import numpy as np
from scipy.spatial.distance import cityblock

from settings import COLS, ROWS


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
    self.model_number = None
    self.Q = None

    try:
        with open("model.pt", "rb") as file:
            self.Q = pickle.load(file)
        with open("modelList.txt", "r") as model_file:
            model_list = model_file.readline().split(",")
            tmp2 = model_list.pop(0)
            model_list.append(tmp2)
            self.model_number = tmp2
        with open("modelList.txt", "w") as model_file:
            model_file.write(",".join(model_list))

    except (EOFError, FileNotFoundError):
        try:
            with open("modelList.txt", "r") as model_file:
                model_list = model_file.readline().split(",")
                tmp2 = model_list.pop(0)
                model_list.append(tmp2)
                self.model_number = tmp2
            with open("modelList.txt", "w") as model_file:
                model_file.write(",".join(model_list))
            with open(f"model{self.model_number}", "rb") as file:
                self.Q = pickle.load(file)
                print(f"Loaded: {self.model_number}")
        except (EOFError, FileNotFoundError):
            if self.Q is None:
                self.Q = np.zeros((6, 2, 6, 5, 6, 6))

    if np.sum(self.Q) != 0:
        print("Loaded")


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 50), 0.1)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

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


def get_crate_positions(game_state):
    return np.argwhere(game_state["field"] == 1)


def get_other_agent_positions(game_state):
    return np.array([x[3] for x in game_state["others"]])


def get_idx_for_state(game_state: dict):
    agent_position = np.array(game_state["self"][3])
    crate_positions = get_crate_positions(game_state)
    coin_positions = game_state["coins"]
    other_agent_positions = get_other_agent_positions(game_state)

    game_map = map_game_state_to_image(game_state)

    safe_field_direction_idx, _ = find_next_safe_field(game_map, agent_position)

    can_drop_bomb_idx = 1 if can_drop_bomb(game_state) else 0

    crate_direction_idx, _ = find_next_crate(game_map, agent_position, crate_positions)

    coin_direction_idx, _ = find_next_coin(game_map, agent_position, coin_positions)

    nearest_agent_idx, _ = find_next_agent(game_map, agent_position, other_agent_positions)

    return safe_field_direction_idx, can_drop_bomb_idx, crate_direction_idx, coin_direction_idx, nearest_agent_idx


def can_drop_bomb(game_state):
    return game_state["self"][2]


def is_in_bomb_range(agent_position, bomb_positions):
    return len(bomb_positions) > 0 and len(objects_in_bomb_dist(agent_position, bomb_positions)) > 0


def find_direction_to_increase_crates_destroyed(game_state):
    agent_position = game_state["self"][3]
    crate_positions = get_crate_positions(game_state)
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


def get_movement_indices(agent_position, game_map):
    col, row = agent_position
    top_pos = (col, row - 1)
    right_pos = (col + 1, row)
    bottom_pos = (col, row + 1)
    left_pos = (col - 1, row)

    top = 1 if game_map[top_pos] != 1 else 0
    right = 2 if game_map[right_pos] != 1 else 0
    bottom = 4 if game_map[bottom_pos] != 1 else 0
    left = 8 if game_map[left_pos] != 1 else 0

    return left & right & top & bottom


def map_game_state_to_image(game_state):
    field = (game_state["field"]).copy()  # 0: free tiles, 1: crates, -1: stone walls
    field[field == -1] = 1  # stone walls

    field[game_state["explosion_map"] >= 1] = 1  # explosions

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

    for other_agent in game_state["others"]:
        # we can not move to a field where another player stands
        field[other_agent[3][0], other_agent[3][1]] = 1

    return field  # 0: secure, 1: not passable, 2+: passable, but there will be an explosion in the future


def find_next_safe_field(game_map, agent_position, allowed_steps: int = 4) -> Optional[Tuple[int, int]]:
    """
    Returns: next_step_direction, needed_steps
    """
    # TODO: currently assuming, that we have only one bomb.
    #  With more we have to take the different countdowns into account

    secure_fields = np.argwhere(game_map == 0)

    steps = np.array(get_steps_between(agent_position, np.argwhere(game_map == 0)))

    reachable_within_allowed_steps = secure_fields[steps <= allowed_steps]
    steps = steps[steps <= allowed_steps]

    sorted_indices = np.argsort(steps)

    for idx in sorted_indices:
        is_reachable, next_step_direction, needed_steps = \
            reachable(game_map, reachable_within_allowed_steps[idx], agent_position, limit=allowed_steps)
        if is_reachable:
            return next_step_direction, needed_steps

    return SafeFieldDirection.UNREACHABLE, np.inf


def find_next_crate(game_map, agent_position, crate_positions) -> Optional[Tuple[int, int]]:
    """
    Returns: next_step_direction, needed_steps
    """

    if not len(crate_positions):
        return CrateDirection.UNREACHABLE_OR_NONE, np.inf

    steps = get_steps_between(agent_position, crate_positions)

    sorted_indices = np.argsort(steps)

    if steps[sorted_indices[0]] == 1:
        return CrateDirection.NEXT_TO, 0

    for idx in sorted_indices:
        game_map[crate_positions[idx][0], crate_positions[idx][1]] = 0
        is_reachable, next_step_direction, needed_steps = reachable(game_map, crate_positions[idx], agent_position, limit=30)
        game_map[crate_positions[idx][0], crate_positions[idx][1]] = 1
        if is_reachable:
            return next_step_direction, needed_steps-1
    return CrateDirection.UNREACHABLE_OR_NONE, np.inf


def find_next_coin(game_map, agent_position, coin_positions) -> Optional[Tuple[int, int]]:
    """
    Returns: next_step_direction, needed_steps
    """

    if not len(coin_positions):
        return CoinDirection.UNREACHABLE_OR_NONE, np.inf

    steps = get_steps_between(agent_position, coin_positions)

    sorted_indices = np.argsort(steps)

    for idx in sorted_indices:
        is_reachable, next_step_direction, needed_steps = reachable(game_map, coin_positions[idx], agent_position, limit=100)
        if is_reachable:
            return next_step_direction, needed_steps
    return CoinDirection.UNREACHABLE_OR_NONE, np.inf


def find_next_agent(game_map, agent_position, other_agent_positions) -> Optional[Tuple[int, int]]:
    """
    Returns: next_step_direction, needed_steps
    """

    if not len(other_agent_positions):
        return NearestAgentDirection.UNREACHABLE_OR_NONE, np.inf

    agents_in_bomb_range = objects_in_bomb_dist(agent_position, other_agent_positions)

    if len(agents_in_bomb_range) > 0:
        return NearestAgentDirection.IN_BOMB_RANGE, 0

    steps = get_steps_between(agent_position, other_agent_positions)

    sorted_indices = np.argsort(steps)

    for idx in sorted_indices:
        game_map[other_agent_positions[idx][0], other_agent_positions[idx][1]] = 0
        is_reachable, next_step_direction, needed_steps = reachable(game_map, other_agent_positions[idx], agent_position, limit=30)
        game_map[other_agent_positions[idx][0], other_agent_positions[idx][1]] = 1
        if is_reachable:
            return next_step_direction, needed_steps - 1
    return NearestAgentDirection.UNREACHABLE_OR_NONE, np.inf


def reachable(game_map, pos, agent_position, step=0, limit=4) -> Tuple[bool, int, int]:
    sign_x, sign_y = np.sign(pos - agent_position)
    diff_x, diff_y = np.abs(pos - agent_position)

    if step > limit:
        return False, Direction.UNREACHABLE, np.inf

    if sign_x == 0 and sign_y == 0:
        return True, Direction.IS_AT, step

    if sign_x != 0:
        # DIFFERENT COLUMN
        if diff_x == 1 and game_map[agent_position[0] + sign_x, agent_position[1] + sign_y] == 1:
            # IF THERE'S A BLOCK ON OUR TOP/BOTTOM DIAGONAL, WE DON'T GO LEFT OR RIGHT BUT UP/DOWN
            pass
        elif game_map[agent_position[0] + sign_x, agent_position[1]] != 1:
            # GO LEFT/RIGHT IN SAME ROW
            ret = reachable(game_map, pos, np.array([agent_position[0] + sign_x, agent_position[1]]), step+1, limit)
            if ret[0]:
                return True, Direction.RIGHT if sign_x > 0 else Direction.LEFT, ret[2]
    elif game_map[agent_position[0], agent_position[1] + sign_y] == 1:
        # SAME COLUMN, FIELD ABOVE/BELOW IS BLOCKED
        if game_map[agent_position[0] - 1, agent_position[1]] != 1 and game_map[agent_position[0] - 1, agent_position[1] + sign_y] != 1:
            # FIELD LEFT AND LEFT UP/DOWN IS FREE --> GO LEFT
            ret = reachable(game_map, pos, np.array([agent_position[0] - 1, agent_position[1] + sign_y]), step + 2, limit)
            if ret[0]:
                return True, Direction.LEFT, ret[2]  # go left
        if game_map[agent_position[0] + 1, agent_position[1]] != 1 and game_map[agent_position[0] + 1, agent_position[1] + sign_y] != 1:
            # FIELD RIGHT AND RIGHT UP/DOWN IS FREE --> GO RIGHT
            ret = reachable(game_map, pos, np.array([agent_position[0] + 1, agent_position[1] + sign_y]), step + 2, limit)
            if ret[0]:
                return True, Direction.RIGHT, ret[2]  # go right

    if sign_y != 0:
        # DIFFERENT ROW
        if diff_y == 1 and game_map[agent_position[0] + sign_x, agent_position[1] + sign_y] == 1:
            # IF THERE'S A BLOCK ON OUR LEFT/RIGHT DIAGONAL, WE DON'T GO UP/DOWN BUT RIGHT/LEFT
            pass
        elif game_map[agent_position[0], agent_position[1] + sign_y] != 1:
            # GO UP/DOWN IN SAME COLUMN
            ret = reachable(game_map, pos, np.array([agent_position[0], agent_position[1] + sign_y]), step + 1, limit)
            if ret[0]:
                return True, Direction.UP if sign_y < 0 else Direction.DOWN, ret[2]
    elif game_map[agent_position[0] + sign_x, agent_position[1]] == 1:
        # SAME ROW, FIELD RIGHT/LEFT IS BLOCKED
        if game_map[agent_position[0], agent_position[1] - 1] != 1 and game_map[agent_position[0] + sign_x, agent_position[1] - 1] != 1:
            # FIELD UP AND UP RIGHT/LEFT IS FREE --> GO UP
            ret = reachable(game_map, pos, np.array([agent_position[0] + sign_x, agent_position[1] - 1]), step + 2, limit)
            if ret[0]:
                return True, Direction.UP, ret[2]  # go up
        if game_map[agent_position[0], agent_position[1] + 1] != 1 and game_map[agent_position[0] + sign_x, agent_position[1] + 1] != 1:
            # FIELD BELOW AND BELOW RIGHT/LEFT IS FREE --> GO DOWN
            ret = reachable(game_map, pos, np.array([agent_position[0] + sign_x, agent_position[1] + 1]), step + 2, limit)
            if ret[0]:
                return True, Direction.DOWN, ret[2]  # go down

    return False, Direction.UNREACHABLE, np.inf
