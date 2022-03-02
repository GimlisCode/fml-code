from collections import namedtuple, deque

from typing import List

import matplotlib.pyplot as plt

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

    # State: (-x-y, -x0y, -x+y, 0x-y, 0x-0y, 0x+y, +x-y, +x0y, +x+y) * 3 for next to blocks left/right/not -> 27
    # Actions: 'UP', 'RIGHT', 'DOWN', 'LEFT' -> 4

    try:
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)
    except (EOFError, FileNotFoundError):
        self.Q = np.random.rand(9, 3, 3, 3, 4) * 3

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
        self.transitions.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), 0))
        return

    reward = calculate_reward(events, old_game_state, new_game_state)

    state_idx_w_t, state_idx_x_t, state_idx_y_t, state_idx_z_t = get_idx_for_state(old_game_state)
    state_idx_w_t1, state_idx_x_t1, state_idx_y_t1, state_idx_z_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    self.Q[state_idx_w_t, state_idx_x_t, state_idx_y_t, state_idx_z_t, action_idx_t] += self.alpha * (reward + self.gamma * np.max(self.Q[state_idx_w_t1, state_idx_x_t1, state_idx_y_t1, state_idx_z_t1]) - self.Q[state_idx_w_t, state_idx_x_t, state_idx_y_t, state_idx_z_t, action_idx_t])

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), calculate_reward(events, old_game_state, new_game_state)))


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
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q, file)

    # plt.gray()
    # plt.imshow(np.reshape(self.Q, (9 * 27, 4)))
    # plt.show()


def calculate_reward(events, old_game_state, new_game_state) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -10

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_nearest_coin_dist_x, previous_nearest_coin_dist_y = state_to_features(old_game_state)
    nearest_coin_dist_x, nearest_coin_dist_y = state_to_features(new_game_state)

    if nearest_coin_dist_x < previous_nearest_coin_dist_x or nearest_coin_dist_y < previous_nearest_coin_dist_y:
        reward_sum += 3
    else:
        reward_sum -= 1

    return reward_sum
