from typing import List

import numpy as np

import events as e
from .callbacks import *


def setup_training(self):
    self.Q.learning_rate = 0.00001
    self.Q.gamma = 0.25

    self.optimizer = self.Q.configure_optimizers()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    if len(np.array(new_game_state["coins"])) == 0:
        return

    reward = calculate_reward(events, old_game_state, new_game_state)

    state_features_t = state_to_features(old_game_state).to(self.device)

    state_features_t_plus_1 = state_to_features(new_game_state).to(self.device)

    action = get_idx_for_action(self_action)

    self.Q.train()

    self.optimizer.zero_grad()
    loss = self.Q.training_step((state_features_t, action, reward, state_features_t_plus_1), 0)
    loss.backward()
    self.optimizer.step()

    self.train_info["train_loss"].append(loss.item())

    self.Q.eval()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.Q.save("model.pt")

    with open("train_info.json", "w") as file:
        file.write(json.dumps(self.train_info))


def calculate_reward(events, old_game_state, new_game_state) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -1

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_min_dist = np.min(get_steps_between(np.array(old_game_state["self"][3]), np.array(old_game_state["coins"])))
    current_min_dist = np.min(get_steps_between(np.array(new_game_state["self"][3]), np.array(new_game_state["coins"])))

    if current_min_dist < previous_min_dist:
        reward_sum += 0.4
    else:
        reward_sum -= 0.5

    return np.clip(reward_sum, a_min=-1, a_max=1)
