import random

import numpy as np
import torch

from .network import QNetwork


MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    self.Q = QNetwork(features_out=4)
    self.Q.load("model.pt")
    self.Q.eval()


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]

    random_prob = max(.5**(1 + current_round / 15), 0.25)
    if self.train and random.random() < random_prob:
        return np.random.choice(MOVE_ACTIONS)

    features = map_game_state_to_image(game_state)
    return MOVE_ACTIONS[torch.argmax(self.Q.forward(features)[0, :])]


def map_game_state_to_image(game_state):
    field = game_state["field"]  # 0: free tiles, 1: crates, -1: stone walls

    field[field == -1] = 2  # 2: stone walls

    field[game_state["self"][3]] = 3  # 3: player

    for coin in game_state["coins"]:
        field[coin] = 4  # 4: coin

    img = torch.zeros((1, 1, 18, 18), dtype=torch.double)
    img[0, 0, 0:-1, 0:-1] = torch.tensor(field) / 4

    return img
