from collections import deque

from typing import List


import events as e
from .callbacks import *

direction_mapping = {
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
    4: "UP"
}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
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

    reward = calculate_reward(events, old_game_state, new_game_state, self_action)

    idx_t = get_idx_for_state(old_game_state)
    idx_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    self.Q[idx_t][action_idx_t] += self.alpha * (
                reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx_t])


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

    # idx_t = get_idx_for_state(last_game_state)
    # action_idx_t = get_idx_for_action(last_action)

    # if e.KILLED_SELF in events:
    #     self.Q[idx_t][action_idx_t] += self.alpha * (-20 - self.Q[idx_t][action_idx_t])

    # Store the model
    # Store the model
    if self.model_number is not None:
        with open(f"model{self.model_number}.pt", "wb") as file:
            pickle.dump(self.Q, file)
    else:
        with open(f"model.pt", "wb") as file:
            pickle.dump(self.Q, file)


def calculate_reward(events, old_game_state, new_game_state, action) -> int:
    game_rewards = {
        e.INVALID_ACTION: -2,
        e.COIN_COLLECTED: 3
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_agent_position = np.array(old_game_state["self"][3])
    current_agent_position = np.array(new_game_state["self"][3])

    previous_crate_positions = extract_crate_positions(old_game_state["field"])
    current_crate_positions = extract_crate_positions(new_game_state["field"])

    previous_coin_positions = old_game_state["coins"]
    current_coin_positions = new_game_state["coins"]

    previous_map = map_game_state_to_image(old_game_state)
    current_map = map_game_state_to_image(new_game_state)

    previous_crate_direction, previous_crate_distance = find_next_crate(previous_map, previous_agent_position, previous_crate_positions)
    current_crate_direction, current_crate_distance = find_next_crate(current_map, current_agent_position, current_crate_positions)

    # --- BOMB DROP ---
    if e.BOMB_DROPPED in events and previous_crate_distance == CrateDirection.NEXT_TO:
        # AGENT DROPPED A BOMB AND WAS NEXT TO A CRATE -> good
        reward_sum += 1

    # --- BOMB DODGE ---
    previous_safe_field_direction, previous_safe_field_distance = find_next_safe_field(previous_map, previous_agent_position)
    current_safe_field_direction, current_safe_field_distance = find_next_safe_field(current_map, current_agent_position)
    if previous_safe_field_distance > current_safe_field_distance and current_safe_field_direction != SafeFieldDirection.UNREACHABLE:
        # AGENT MOVED CLOSER TO SAFE FIELD
        reward_sum += 4
    elif current_safe_field_direction != SafeFieldDirection.IS_AT and e.BOMB_DROPPED not in events:
        # AGENT DID NOT MOVE CLOSER TO SAFE FIELD AND IS NOT AT SAFE FIELD
        reward_sum -= 4

    # --- COINS ---
    previous_coin_direction, previous_coin_distance = find_next_coin(previous_map, previous_agent_position, previous_coin_positions)
    current_coin_direction, current_coin_distance = find_next_coin(current_map, current_agent_position, current_coin_positions)
    if previous_coin_distance > current_coin_distance:
        # AGENT MOVED CLOSER TO COIN
        reward_sum += 3

    # --- CRATES ---
    if previous_crate_distance > current_crate_distance:
        # AGENT MOVED CLOSER TO CRATE
        reward_sum += 2

    return reward_sum
