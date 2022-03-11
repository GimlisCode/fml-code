import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import get_idx_for_state, get_idx_for_action, map_game_state_to_image, find_next_coin

direction_mapping = {
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
    4: "UP"
}


def setup_training(self):
    self.alpha = 0.2
    self.gamma = 0.5


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    reward = calculate_reward(events, old_game_state, new_game_state, self_action)

    idx_t = get_idx_for_state(old_game_state)
    idx_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    self.Q[idx_t][action_idx_t] += self.alpha * (reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx_t])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Store the model
    with open("model.pt", "wb") as file:
        pickle.dump(self.Q, file)


def calculate_reward(events, old_game_state, new_game_state, action) -> int:
    game_rewards = {
        e.INVALID_ACTION: -50
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_agent_position = np.array(old_game_state["self"][3])

    previous_coin_positions = old_game_state["coins"]

    previous_map = map_game_state_to_image(old_game_state)

    # OBJECTIVE: COLLECT COIN(S)
    ret_coins = find_next_coin(previous_map, previous_agent_position, previous_coin_positions)

    if ret_coins is not None:
        # COIN WAS REACHABLE
        previous_coin_direction, previous_steps_to_coin = ret_coins

        if previous_coin_direction != 0:
            # AGENT KNEW COIN DIRECTION
            if direction_mapping[previous_coin_direction] != action:
                # AGENT TOOK OTHER ACTION
                reward_sum -= 20
            else:
                # AGENT FOLLOWED OUR GUIDANCE
                reward_sum += 20

    return reward_sum
