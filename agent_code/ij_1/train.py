from typing import List


import events as e
from .callbacks import *


def setup_training(self):
    self.alpha = 0.2
    self.gamma = 0.5


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    reward = calculate_reward(events, old_game_state, new_game_state)

    idx_t = get_idx_for_state(old_game_state)
    idx_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    self.Q[idx_t][action_idx_t] += self.alpha * (
                reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx_t])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if self.model_number is not None:
        with open(f"model{self.model_number}.pt", "wb") as file:
            pickle.dump(self.Q, file)
    else:
        with open(f"model.pt", "wb") as file:
            pickle.dump(self.Q, file)


def calculate_reward(events, old_game_state, new_game_state) -> int:
    game_rewards = {
        e.INVALID_ACTION: -2,
        e.COIN_COLLECTED: 4
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_agent_position = np.array(old_game_state["self"][3])
    current_agent_position = np.array(new_game_state["self"][3])

    previous_crate_positions = get_crate_positions(old_game_state)
    current_crate_positions = get_crate_positions(new_game_state)

    previous_coin_positions = old_game_state["coins"]
    current_coin_positions = new_game_state["coins"]

    previous_map = map_game_state_to_image(old_game_state)
    current_map = map_game_state_to_image(new_game_state)

    previous_other_agent_positions = get_other_agent_positions(old_game_state)
    current_other_agent_positions = get_other_agent_positions(new_game_state)

    previous_crate_direction, previous_crate_distance = find_next_crate(previous_map, previous_agent_position,
                                                                        previous_crate_positions)
    current_crate_direction, current_crate_distance = find_next_crate(current_map, current_agent_position,
                                                                      current_crate_positions)

    previous_coin_direction, previous_coin_distance = find_next_coin(previous_map, previous_agent_position,
                                                                     previous_coin_positions)
    current_coin_direction, current_coin_distance = find_next_coin(current_map, current_agent_position,
                                                                   current_coin_positions)

    previous_safe_field_direction, previous_safe_field_distance = find_next_safe_field(previous_map,
                                                                                       previous_agent_position)
    current_safe_field_direction, current_safe_field_distance = find_next_safe_field(current_map,
                                                                                     current_agent_position)

    previous_other_agent_direction, previous_other_agent_distance = find_next_agent(previous_map,
                                                                                    previous_agent_position,
                                                                                    previous_other_agent_positions)
    current_other_agent_direction, current_other_agent_distance = find_next_agent(current_map,
                                                                                  current_agent_position,
                                                                                  current_other_agent_positions)

    # --- BOMB DROP ---
    next_to_crate = previous_crate_direction == CrateDirection.NEXT_TO
    in_bomb_range = previous_other_agent_direction == NearestAgentDirection.IN_BOMB_RANGE

    if e.BOMB_DROPPED in events and next_to_crate:
        # AGENT DROPPED A BOMB AND WAS NEXT TO A CRATE
        reward_sum += 1
    if e.BOMB_DROPPED in events and in_bomb_range:
        # AGENT DROPPED A BOMB AND ANOTHER AGENT WAS IN BOMB RANGE
        reward_sum += 1
    if e.BOMB_DROPPED in events and not next_to_crate and not in_bomb_range:
        # AGENT DROPPED A BOMB BUT THERE WAS NO TARGET
        reward_sum -= 2

    # --- BOMB DODGE ---
    if previous_safe_field_distance > current_safe_field_distance and current_safe_field_direction != SafeFieldDirection.UNREACHABLE:
        # AGENT MOVED CLOSER TO SAFE FIELD
        reward_sum += 5
    elif current_safe_field_direction != SafeFieldDirection.IS_AT and e.BOMB_DROPPED not in events:
        # AGENT DID NOT MOVE CLOSER TO SAFE FIELD AND IS NOT AT SAFE FIELD
        reward_sum -= 9  # as the other positive reward can at most sum up to 9 in this case

    # --- COINS ---
    if previous_coin_distance > current_coin_distance:
        # AGENT MOVED CLOSER TO COIN
        reward_sum += 4

    # --- CRATES ---
    if previous_crate_distance > current_crate_distance:
        # AGENT MOVED CLOSER TO CRATE
        reward_sum += 3

    # --- OTHER AGENTS ---
    agent_moved = not (e.BOMB_DROPPED in events or e.WAITED in events or e.INVALID_ACTION in events)
    if previous_other_agent_distance > current_other_agent_distance and agent_moved:
        # AGENT MOVED CLOSER TO THE NEAREST OTHER AGENT
        reward_sum += 2

    return reward_sum
