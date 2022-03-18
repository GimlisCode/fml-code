from enum import Enum

"""
Possible augmentations:
    -> 90 degree rotation
    -> 180 degree rotation
    -> 270 degree rotation
    -> flip left/right
    -> flip up/down
    -> 90 degree rotation + flip left/right
    -> 270 degree rotation + flip left/right
"""


class Augmentation(Enum):
    ROTATE_90_DEGREE = 0
    ROTATE_180_DEGREE = 1
    ROTATE_270_DEGREE = 2
    FLIP_LEFT_RIGHT = 3
    FLIP_UP_DOWN = 4
    ROTATE_90_FLIP_LEFT_RIGHT = 5
    ROTATE_270_FLIP_LEFT_RIGHT = 6


def get_all_augmentations(state_idx_t, action_idx, state_idx_t1):
    augmented_sas = list()

    for augmentation in Augmentation:
        if augmentation in [Augmentation.ROTATE_270_FLIP_LEFT_RIGHT, Augmentation.ROTATE_90_FLIP_LEFT_RIGHT,
                            Augmentation.FLIP_LEFT_RIGHT, Augmentation.FLIP_UP_DOWN]:
            continue
        augmented_sas.append(augment(state_idx_t, action_idx, state_idx_t1, augmentation))

    return augmented_sas


def augment(state_idx_t, action_idx, state_idx_t1, augmentation: Augmentation):
    return (
        augment_state(state_idx_t, augmentation),
        augment_action(action_idx, augmentation),
        augment_state(state_idx_t1, augmentation)
    )


def augment_state(state_idx, augmentation: Augmentation):
    # safe_field_direction_idx, can_drop_bomb_idx, crate_direction_idx, coin_direction_idx, nearest_agent_idx
    if augmentation == Augmentation.ROTATE_90_DEGREE:
        return (
            rotate_direction_90_degree(state_idx[0]),
            state_idx[1],
            rotate_direction_90_degree(state_idx[2]),
            rotate_direction_90_degree(state_idx[3]),
            rotate_direction_90_degree(state_idx[4]),
        )
    if augmentation == Augmentation.ROTATE_180_DEGREE:
        return (
            rotate_direction_180_degree(state_idx[0]),
            state_idx[1],
            rotate_direction_180_degree(state_idx[2]),
            rotate_direction_180_degree(state_idx[3]),
            rotate_direction_180_degree(state_idx[4]),
        )
    if augmentation == Augmentation.ROTATE_270_DEGREE:
        return (
            rotate_direction_270_degree(state_idx[0]),
            state_idx[1],
            rotate_direction_270_degree(state_idx[2]),
            rotate_direction_270_degree(state_idx[3]),
            rotate_direction_270_degree(state_idx[4]),
        )
    if augmentation == Augmentation.FLIP_LEFT_RIGHT:
        return (
            flip_direction_left_right(state_idx[0]),
            state_idx[1],
            flip_direction_left_right(state_idx[2]),
            flip_direction_left_right(state_idx[3]),
            flip_direction_left_right(state_idx[4]),
        )
    if augmentation == Augmentation.FLIP_UP_DOWN:
        return (
            flip_direction_up_down(state_idx[0]),
            state_idx[1],
            flip_direction_up_down(state_idx[2]),
            flip_direction_up_down(state_idx[3]),
            flip_direction_up_down(state_idx[4]),
        )
    if augmentation == Augmentation.ROTATE_90_FLIP_LEFT_RIGHT:
        return (
            rotate_direction_90_and_flip_left_right(state_idx[0]),
            state_idx[1],
            rotate_direction_90_and_flip_left_right(state_idx[2]),
            rotate_direction_90_and_flip_left_right(state_idx[3]),
            rotate_direction_90_and_flip_left_right(state_idx[4]),
        )
    if augmentation == Augmentation.ROTATE_270_FLIP_LEFT_RIGHT:
        return (
            rotate_direction_270_and_flip_left_right(state_idx[0]),
            state_idx[1],
            rotate_direction_270_and_flip_left_right(state_idx[2]),
            rotate_direction_270_and_flip_left_right(state_idx[3]),
            rotate_direction_270_and_flip_left_right(state_idx[4]),
        )
    else:
        raise ValueError(f"The given augmentation {augmentation} is unknown!")


def augment_action(action_idx, augmentation: Augmentation):
    if augmentation == Augmentation.ROTATE_90_DEGREE:
        return rotate_action_90_degree(action_idx)
    if augmentation == Augmentation.ROTATE_180_DEGREE:
        return rotate_action_180_degree(action_idx)
    if augmentation == Augmentation.ROTATE_270_DEGREE:
        return rotate_action_270_degree(action_idx)
    if augmentation == Augmentation.FLIP_LEFT_RIGHT:
        return flip_action_left_right(action_idx)
    if augmentation == Augmentation.FLIP_UP_DOWN:
        return flip_action_up_down(action_idx)
    if augmentation == Augmentation.ROTATE_90_FLIP_LEFT_RIGHT:
        return rotate_action_90_and_flip_left_right(action_idx)
    if augmentation == Augmentation.ROTATE_270_FLIP_LEFT_RIGHT:
        return rotate_action_270_and_flip_left_right(action_idx)
    else:
        raise ValueError(f"The given augmentation {augmentation} is unknown!")


def augment_direction(direction_idx, augmentation_dict):
    if direction_idx in [1, 2, 3, 4]:
        return augmentation_dict[direction_idx]
    return direction_idx


def rotate_direction_90_degree(direction_idx):
    # clockwise
    return augment_direction(direction_idx, {
        1: 2,  # Right --> Down
        2: 3,  # Down --> Left
        3: 4,  # Left --> Up
        4: 1,  # Up --> Right
    })


def rotate_direction_180_degree(direction_idx):
    return rotate_direction_90_degree(rotate_direction_90_degree(direction_idx))


def rotate_direction_270_degree(direction_idx):
    return rotate_direction_90_degree(rotate_direction_180_degree(direction_idx))


def flip_direction_left_right(direction_idx):
    return augment_direction(direction_idx, {
        1: 3,  # Right --> Left
        2: 2,  # Down --> Down
        3: 1,  # Left --> Right
        4: 4,  # Up --> Up
    })


def flip_direction_up_down(direction_idx):
    return augment_direction(direction_idx, {
        1: 1,  # Right --> Right
        2: 4,  # Down --> Up
        3: 3,  # Left --> Left
        4: 2,  # Up --> Down
    })


def rotate_direction_90_and_flip_left_right(direction_idx):
    return flip_direction_left_right(rotate_direction_90_degree(direction_idx))


def rotate_direction_270_and_flip_left_right(direction_idx):
    return flip_direction_left_right(rotate_direction_270_degree(direction_idx))


def augment_action_idx(action_idx, augmentation_dict):
    if action_idx in [0, 1, 2, 3]:
        return augmentation_dict[action_idx]
    return action_idx


def rotate_action_90_degree(action_idx):
    # clockwise
    return augment_action_idx(action_idx, {
        0: 1,  # Up --> Right
        1: 2,  # Right --> Down
        2: 3,  # Down --> Left
        3: 0,  # Left --> Up
    })


def rotate_action_180_degree(action_idx):
    return rotate_action_90_degree(rotate_action_90_degree(action_idx))


def rotate_action_270_degree(action_idx):
    return rotate_action_90_degree(rotate_action_180_degree(action_idx))


def flip_action_left_right(action_idx):
    return augment_action_idx(action_idx, {
        0: 0,  # Up --> Up
        1: 3,  # Right --> Left
        2: 2,  # Down --> Down
        3: 1,  # Left --> Right
    })


def flip_action_up_down(action_idx):
    return augment_action_idx(action_idx, {
        0: 2,  # Up --> Down
        1: 1,  # Right --> Right
        2: 0,  # Down --> Up
        3: 3,  # Left --> Left
    })


def rotate_action_90_and_flip_left_right(action_idx):
    return flip_action_left_right(rotate_action_90_degree(action_idx))


def rotate_action_270_and_flip_left_right(action_idx):
    return flip_action_left_right(rotate_action_270_degree(action_idx))
