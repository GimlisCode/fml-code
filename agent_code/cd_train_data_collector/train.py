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
    self.alpha = 0.2
    self.gamma = 0.5


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    reward = calculate_reward(events, old_game_state, new_game_state, self_action)

    idx_t = get_idx_for_state(old_game_state)
    idx_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    self.Q[idx_t][action_idx_t] += self.alpha * (
                reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx_t])

    self.dataset.add(
        state_idx_to_one_hot_encoding(idx_t),
        action_idx_t,
        reward,
        state_idx_to_one_hot_encoding(idx_t1)
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    with open("model.pt", "wb") as file:
        pickle.dump(self.Q, file)

    self.dataset.save("train_data.json")


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


def state_idx_to_one_hot_encoding(state_idx):
    # safe_field_direction_idx,     can_drop_bomb_idx,  crate_direction_idx,    coin_direction_idx
    # 7,                            2,                  7,                      6

    feature_vector = list()

    # safe_field_direction_idx  -->  one hot encoding for values 0 to 6
    feature_vector.append(1 if state_idx[0] == 0 else 0)
    feature_vector.append(1 if state_idx[0] == 1 else 0)
    feature_vector.append(1 if state_idx[0] == 2 else 0)
    feature_vector.append(1 if state_idx[0] == 3 else 0)
    feature_vector.append(1 if state_idx[0] == 4 else 0)
    feature_vector.append(1 if state_idx[0] == 5 else 0)
    feature_vector.append(1 if state_idx[0] == 6 else 0)

    # can_drop_bomb_idx  -->  already binary
    feature_vector.append(state_idx[1])

    # crate_direction_idx  -->  one hot encoding for values 0 to 6
    feature_vector.append(1 if state_idx[2] == 0 else 0)
    feature_vector.append(1 if state_idx[2] == 1 else 0)
    feature_vector.append(1 if state_idx[2] == 2 else 0)
    feature_vector.append(1 if state_idx[2] == 3 else 0)
    feature_vector.append(1 if state_idx[2] == 4 else 0)
    feature_vector.append(1 if state_idx[2] == 5 else 0)
    feature_vector.append(1 if state_idx[2] == 6 else 0)

    # coin_direction_idx  -->  one hot encoding for values 0 to 5
    feature_vector.append(1 if state_idx[3] == 0 else 0)
    feature_vector.append(1 if state_idx[3] == 1 else 0)
    feature_vector.append(1 if state_idx[3] == 2 else 0)
    feature_vector.append(1 if state_idx[3] == 3 else 0)
    feature_vector.append(1 if state_idx[3] == 4 else 0)
    feature_vector.append(1 if state_idx[3] == 5 else 0)

    return feature_vector



