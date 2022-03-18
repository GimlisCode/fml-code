from typing import List
import json

import numpy as np

from agent_code.ij_1.train import calculate_reward
from agent_code.ij_1.callbacks import map_game_state_to_image, get_idx_for_action


import events as e


def setup_training(self):
    self.rewards_per_epoch = list()
    self.current_points = list()
    self.agent_died = list()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        self.rewards = list()
        return

    reward = calculate_reward(events, old_game_state, new_game_state)

    self.rewards.append(reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    last_game_map = map_game_state_to_image(last_game_state)
    last_agent_position = np.array(last_game_state["self"][3])
    agent_died = e.KILLED_SELF in events or e.GOT_KILLED in events

    if agent_died and last_game_map[last_agent_position[0], last_agent_position[1]] == 0:
        # THE AGENT DIED AND WAS ON A SAVE FIELD BEFORE HE MOVED
        action_idx = get_idx_for_action(last_action)
        if action_idx < 4:  # < 4 means he moved somehow into his death
            # IF HE ACTIVELY MOVED INTO AN EXPLOSION UPDATE Q WITH A NEGATIVE REWARD
            self.rewards.append(-5)

    self.rewards_per_epoch.append(self.rewards)
    self.current_points.append(last_game_state["self"][1])
    self.agent_died.append(1 if agent_died else 0)

    if last_game_state["round"] % 10 == 0:
        with open("performance.json", "w") as f:
            f.write(json.dumps({
                "rewards": self.rewards_per_epoch,
                "points": self.current_points,
                "agent_died": self.agent_died
            }))
