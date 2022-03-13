from typing import List
from pathlib import Path

import numpy as np
import tifffile

import events as e
from .callbacks import *
from .augmentations import get_all_augmentations


def setup_training(self):
    self.alpha = 0.2
    self.gamma = 0.5

    self.save_model_every_k_steps = 50

    self.do_augmentations = True

    self.save_snapshots = False
    self.snap_shot_every_k_steps = 10
    self.snapshot_folder = Path("snapshots")

    if self.model_number is not None:
        self.snapshot_folder = self.snapshot_folder.joinpath(self.model_number)

    if self.save_snapshots and not self.snapshot_folder.exists():
        try:
            self.snapshot_folder.mkdir()
        except FileNotFoundError:
            self.snapshot_folder.parent.mkdir()
            self.snapshot_folder.mkdir()
        self.snapshot_idx = self.snap_shot_every_k_steps
    elif self.save_snapshots:
        snapshot_numbers = [int(f.name.replace(".pt", "").split("_")[1]) for f in self.snapshot_folder.glob("*.pt")]
        self.snapshot_idx = max(snapshot_numbers) if len(snapshot_numbers) > 0 else 0 + self.snap_shot_every_k_steps
    else:
        self.snapshot_idx = self.snap_shot_every_k_steps


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    reward = calculate_reward(events, old_game_state, new_game_state)

    idx_t = get_idx_for_state(old_game_state)
    idx_t1 = get_idx_for_state(new_game_state)
    action_idx_t = get_idx_for_action(self_action)

    update_Q(self, idx_t, action_idx_t, reward, idx_t1)

    if self.do_augmentations:
        for idx_t_augmented, action_idx_t_augmented, idx_t1_augmented in get_all_augmentations(idx_t, action_idx_t, idx_t1):
            update_Q(self, idx_t_augmented, action_idx_t_augmented, reward, idx_t1_augmented)

    features_t = get_idx_for_state(old_game_state)
    features_t1 = get_idx_for_state(new_game_state)

    train_element_meta_info = TrainDataPoint(features_t, action_idx_t, reward, features_t1)

    if train_element_meta_info not in self.meta_info:
        old_game_state_img = map_game_state_to_multichannel_image(old_game_state)
        new_game_state_img = map_game_state_to_multichannel_image(new_game_state)

        img = np.concatenate((old_game_state_img, new_game_state_img))
        tifffile.imsave(f"train_data/{self.img_idx}_{get_idx_for_action(self_action)}_{reward}.tif", img)
        self.img_idx += 1
        self.meta_info.append(train_element_meta_info)


def update_Q(self, idx_t, action_idx, reward, idx_t1):
    self.Q[idx_t][action_idx] += self.alpha * (reward + self.gamma * np.max(self.Q[idx_t1]) - self.Q[idx_t][action_idx])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state["round"] % self.save_model_every_k_steps == 0:
        with open(f"model{self.model_number if self.model_number is not None else ''}.pt", "wb") as file:
            pickle.dump(self.Q, file)

    if self.save_snapshots and last_game_state["round"] % self.snap_shot_every_k_steps == 0:
        with open(self.snapshot_folder.joinpath(f"model_{self.snapshot_idx}.pt"), "wb") as file:
            pickle.dump(self.Q, file)
        self.snapshot_idx += self.snap_shot_every_k_steps


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
        reward_sum -= 5

    # --- BOMB DODGE ---
    if previous_safe_field_distance > current_safe_field_distance and current_safe_field_direction != SafeFieldDirection.UNREACHABLE:
        # AGENT MOVED CLOSER TO SAFE FIELD
        reward_sum += 5
    elif current_safe_field_direction != SafeFieldDirection.IS_AT and e.BOMB_DROPPED not in events:
        # AGENT DID NOT MOVE CLOSER TO SAFE FIELD AND IS NOT AT SAFE FIELD
        reward_sum -= 18    # as the other positive reward can at most sum up to 9 but combined with alpha 0,5 the
        # punishment must be greater than 2 * 9 such that it is not less than 9

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

    channel_explosions = np.zeros_like(map)
    channel_explosions[game_state["explosion_map"] >= 1] = 1

    channel_more_infos = np.zeros_like(map)
    channel_more_infos[0, 0] = 1 if can_drop_bomb(game_state) else 0

    return np.stack((
        channel_free_tiles,
        channel_walls,
        channel_player,
        channel_coins,
        channel_crates,
        channel_other_players,
        channel_explosions,
        channel_more_infos
    ))
