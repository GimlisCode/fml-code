import numpy as np
import torch

from .network import QNetwork


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    self.Q = QNetwork(features_out=6)
    self.Q.load_from_pl_checkpoint("final_epoch.ckpt")
    self.Q.eval()


def act(self, game_state: dict) -> str:
    features = map_game_state_to_multichannel_image(game_state)
    return ACTIONS[torch.argmax(self.Q.forward(features)[0, :])]


def can_drop_bomb(game_state):
    return game_state["self"][2]


def get_other_agent_positions(game_state):
    return np.array([x[3] for x in game_state["others"]])


def map_game_state_to_multichannel_image(game_state):
    map = game_state["field"]  # 0: free tiles, 1: crates, -1: stone walls

    channel_free_tiles = np.zeros_like(map)
    channel_free_tiles[map == 0] = 1

    channel_walls = np.zeros_like(map)
    channel_walls[map == -1] = 1

    channel_player = np.zeros_like(map)
    channel_player[game_state["self"][3]] = 1

    channel_coins = np.zeros_like(map)
    for coin in game_state["coins"]:
        channel_coins[coin] = 1

    channel_crates = np.zeros_like(map)
    channel_crates[map == 1] = 1

    channel_other_players = np.zeros_like(map)
    for other_agent in get_other_agent_positions(game_state):
        channel_other_players[other_agent[0], other_agent[1]] = 1

    channel_bombs = np.zeros_like(map)
    for bomb_pos, bomb_countdown in game_state["bombs"]:
        # giving countdown 0 (exploding next step) the 1
        # and countdowns 1-3 values below 1 but greater 0 (therefore / 3.1)
        channel_bombs[bomb_pos[0], bomb_pos[1]] = 1 - (bomb_countdown / 3.1)

    channel_explosions = np.zeros_like(map)
    channel_explosions[game_state["explosion_map"] >= 1] = 1

    channel_more_infos = np.zeros_like(map)
    channel_more_infos[0, 0] = 1 if can_drop_bomb(game_state) else 0

    img = torch.zeros((1, 9, 18, 18), dtype=torch.double)
    img[0, :, 0:-1, 0:-1] = torch.tensor(np.stack((
        channel_free_tiles,
        channel_walls,
        channel_player,
        channel_coins,
        channel_crates,
        channel_other_players,
        channel_bombs,
        channel_explosions,
        channel_more_infos
    )))

    return img
