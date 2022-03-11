import torch


def rotate_90_degree_anticlockwise(img_t: torch.Tensor, action: torch.Tensor, img_t1: torch.Tensor):
    # rotate the transposed matrix anticlockwise, because the game field is the transposed matrix
    # however, this results in a clockwise rotated matrix

    img_t_new = torch.transpose(torch.rot90(torch.transpose(img_t, 1, 2), k=1, dims=[1, 2]), 1, 2)
    img_t1_new = torch.transpose(torch.rot90(torch.transpose(img_t1, 1, 2), k=1, dims=[1, 2]), 1, 2)

    action_new = {
        0: 3,  # Up -> Left
        1: 0,  # Right -> Up
        2: 1,  # Down -> Right
        3: 2,  # Left -> Down
    }[action.item()]

    return img_t_new, torch.tensor(action_new), img_t1_new


def rotate_90_degree_clockwise(img_t: torch.Tensor, action: torch.Tensor, img_t1: torch.Tensor):
    img_t_new = torch.transpose(torch.rot90(torch.transpose(img_t, 1, 2), k=3, dims=[1, 2]), 1, 2)
    img_t1_new = torch.transpose(torch.rot90(torch.transpose(img_t1, 1, 2), k=3, dims=[1, 2]), 1, 2)

    action_new = {
        0: 1,  # Up -> Right
        1: 2,  # Right -> Down
        2: 3,  # Down -> Left
        3: 0,  # Left -> Up
    }[action.item()]

    return img_t_new, torch.tensor(action_new), img_t1_new


def rotate_180_degree(img_t: torch.Tensor, action: torch.Tensor, img_t1: torch.Tensor):
    img_t_new = torch.transpose(torch.rot90(torch.transpose(img_t, 1, 2), k=2, dims=[1, 2]), 1, 2)
    img_t1_new = torch.transpose(torch.rot90(torch.transpose(img_t1, 1, 2), k=2, dims=[1, 2]), 1, 2)

    action_new = {
        0: 2,  # Up ->  Down
        1: 3,  # Right -> Left
        2: 0,  # Down -> Up
        3: 1,  # Left -> Right
    }[action.item()]

    return img_t_new, torch.tensor(action_new), img_t1_new


def flip_left_right(img_t: torch.Tensor, action: torch.Tensor, img_t1: torch.Tensor):
    img_t_new = torch.transpose(torch.flip(torch.transpose(img_t, 1, 2), [2]), 1, 2)
    img_t1_new = torch.transpose(torch.flip(torch.transpose(img_t1, 1, 2), [2]), 1, 2)

    action_new = {
        0: 0,  # Up -> Up
        1: 3,  # Right -> Left
        2: 2,  # Down -> Down
        3: 1,  # Left -> Right
    }[action.item()]

    return img_t_new, torch.tensor(action_new), img_t1_new


def flip_up_down(img_t: torch.Tensor, action: torch.Tensor, img_t1: torch.Tensor):
    img_t_new = torch.transpose(torch.flip(torch.transpose(img_t, 1, 2), [1]), 1, 2)
    img_t1_new = torch.transpose(torch.flip(torch.transpose(img_t1, 1, 2), [1]), 1, 2)

    action_new = {
        0: 2,  # Up -> Down
        1: 1,  # Right -> Right
        2: 0,  # Down -> Up
        3: 3,  # Left -> Left
    }[action.item()]

    return img_t_new, torch.tensor(action_new), img_t1_new
