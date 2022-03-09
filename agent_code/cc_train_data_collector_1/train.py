from collections import namedtuple, deque

from typing import List

import numpy as np
import tifffile

import events as e
from .callbacks import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.alpha = 0.2
    self.gamma = 0.5


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is None:
        return

    if len(np.array(new_game_state["coins"])) == 0:
        return

    reward = calculate_reward(events, old_game_state, new_game_state)

    state_idx_t = get_idx_for_state(old_game_state)
    state_idx_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    self.Q[state_idx_t][action_idx_t] += self.alpha * (reward + self.gamma * np.max(self.Q[state_idx_t1]) - self.Q[state_idx_t][action_idx_t])

    features_t = get_features_for_state(old_game_state)
    features_t1 = get_features_for_state(new_game_state)

    train_element_meta_info = TrainDataPoint(features_t, action_idx_t, reward, features_t1)

    if train_element_meta_info not in self.meta_info:
        old_game_state_img = map_game_state_to_multichannel_image(old_game_state)
        new_game_state_img = map_game_state_to_multichannel_image(new_game_state)

        img = np.concatenate((old_game_state_img, new_game_state_img))
        tifffile.imsave(f"train_data_new/{self.img_idx}_{get_idx_for_action(self_action)}_{reward}.tif", img)
        self.img_idx += 1
        self.meta_info.append(train_element_meta_info)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, calculate_reward(events)))

    # Store the model
    with open("model.pt", "wb") as file:
        pickle.dump(self.Q, file)

    with open("train_data_new/meta_info.json", "w") as file:
        file.write(json.dumps({"meta_info": [x.as_dict() for x in self.meta_info]}))

    # plt.gray()
    # plt.imshow(np.reshape(self.Q, (9 * 27, 4)))
    # plt.show()


def calculate_reward(events, old_game_state, new_game_state) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -10

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_min_dist = np.min(get_steps_between(np.array(old_game_state["self"][3]), np.array(old_game_state["coins"])))
    current_min_dist = np.min(get_steps_between(np.array(new_game_state["self"][3]), np.array(new_game_state["coins"])))

    if current_min_dist < previous_min_dist:
        reward_sum += 5
    elif e.COIN_COLLECTED not in events and e.INVALID_ACTION not in events:
        reward_sum -= 5

    return reward_sum


def map_game_state_to_image(game_state):
    field = game_state["field"]  # 0: free tiles, 1: crates, -1: stone walls

    field[field == -1] = 2  # 2: stone walls

    field[game_state["self"][3]] = 3  # 3: player

    for coin in game_state["coins"]:
        field[coin] = 4  # 4: coin

    return field


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

    return np.stack((channel_free_tiles, channel_walls, channel_player, channel_coins))
