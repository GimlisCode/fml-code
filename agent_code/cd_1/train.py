from collections import namedtuple, deque

from typing import List


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
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.alpha = 0.2
    # self.alpha = 0.5
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
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, calculate_reward(events)))

    idx_t = get_idx_for_state(last_game_state)
    action_idx_t = get_idx_for_action(last_action)

    if e.KILLED_SELF in events:
        self.Q[idx_t][action_idx_t] += self.alpha * (-20 - self.Q[idx_t][action_idx_t])

    # Store the model
    if self.model_number is not None:
        with open(f"model{self.model_number}.pt", "wb") as file:
            pickle.dump(self.Q, file)
    else:
        with open(f"model.pt", "wb") as file:
            pickle.dump(self.Q, file)

# def calculate_reward_old(events, old_game_state, new_game_state) -> int:
#     game_rewards = {
#         e.MOVED_LEFT: 1,
#         e.MOVED_RIGHT: 1,
#         e.MOVED_UP: 1,
#         e.MOVED_DOWN: 1,
#         e.WAITED: -1,
#         e.INVALID_ACTION: -15
#     }
#     reward_sum = 0
#     for event in events:
#         if event in game_rewards:
#             reward_sum += game_rewards[event]
#
#     previous_agent_position = np.array(old_game_state["self"][3])
#     current_agent_position = np.array(new_game_state["self"][3])
#
#     previous_bomb_positions = np.array([coords for coords, _ in old_game_state["bombs"]])
#     current_bomb_positions = np.array([coords for coords, _ in new_game_state["bombs"]])
#
#     previous_crate_positions = extract_crate_positions(old_game_state["field"])
#     current_crate_positions = extract_crate_positions(new_game_state["field"])
#
#     if e.BOMB_DROPPED in events:
#         if min(get_steps_between(previous_agent_position, previous_crate_positions)) == 1:
#             reward_sum += 15
#         else:
#             reward_sum -= 15
#     else:
#         if len(current_bomb_positions):
#             # there were and are bombs -> objective: dodge bombs
#             if not is_in_bomb_range(current_agent_position, current_bomb_positions):
#                 # is not in range -> good
#                 reward_sum += 10
#             elif is_in_bomb_range(previous_agent_position, previous_bomb_positions) and min(
#                     get_steps_between(previous_agent_position, previous_bomb_positions)) < min(
#                 get_steps_between(current_agent_position, current_bomb_positions)):
#                 # was in range and still is, but distance to bomb increased -> good
#                 reward_sum += 10
#             else:
#                 # is in range and was not before/distance did not increase -> bad
#                 reward_sum -= 10
#         # elif len(previous_bomb_positions):
#         #     # there were bombs, but they did explode and agent survived --> good
#         #     reward_sum += 10
#         elif not len(previous_bomb_positions):
#             # there were and are no bombs -> objective: move closer to crate
#             if min(get_steps_between(previous_agent_position, previous_crate_positions)) > min(
#                     get_steps_between(current_agent_position, current_crate_positions)):
#                 # moved closer to crate -> good
#                 reward_sum += 10
#             else:
#                 # did not move closer to crate
#                 reward_sum -= 10
#
#     return reward_sum


def calculate_reward(events, old_game_state, new_game_state, action) -> int:
    # game_rewards = {
    #     # e.INVALID_ACTION: -50
    #     # e.COIN_COLLECTED: 20
    # }
    reward_sum = 0
    # for event in events:
    #     if event in game_rewards:
    #         reward_sum += game_rewards[event]

    previous_agent_position = np.array(old_game_state["self"][3])
    # current_agent_position = np.array(new_game_state["self"][3])

    # previous_bomb_positions = np.array([coords for coords, _ in old_game_state["bombs"]])
    # current_bomb_positions = np.array([coords for coords, _ in new_game_state["bombs"]])

    previous_crate_positions = extract_crate_positions(old_game_state["field"])
    # current_crate_positions = extract_crate_positions(new_game_state["field"])

    previous_coin_positions = old_game_state["coins"]
    # current_coin_positions = new_game_state["coins"]

    # previous_explosion_map = old_game_state["explosion_map"]
    # current_explosion_map = new_game_state["explosion_map"]

    previous_map = map_game_state_to_image(old_game_state)
    # current_map = map_game_state_to_image(new_game_state)

    # THIS IS ALREADY PUNISHED BY NOT MOVING TOWARDS COIN/CRATE
    # if not len(previous_bomb_positions) and e.WAITED in events:
    #     # NO BOMB ON THE FIELD
    #     reward_sum -= 50
    if e.BOMB_DROPPED in events:
        # AGENT DROPPED A BOMB -> CHECK IF THERE ARE CRATES/IF THERE WAS A COIN/IF AGENT WAS NEXT TO CRATE

        # if is_in_corner(previous_agent_position):
        #     reward_sum -= 30

        ret_crate = find_next_crate(previous_map, previous_agent_position, previous_crate_positions)

        if ret_crate is None:
            # NO CRATE IS ON THE FIELD
            reward_sum -= 20
        else:
            # THERE ARE STILL CRATES ON THE FIELD
            previous_crate_direction, previous_steps_to_next_crate = ret_crate

            # if previous_crate_direction != 0:
            #     # AGENT KNEW COIN DIRECTION
            #     if direction_mapping[previous_crate_direction] != action:
            #         # AGENT TOOK OTHER ACTION
            #         reward_sum -= 20
            #     else:
            #         # AGENT FOLLOWED OUR GUIDANCE
            #         reward_sum += 20



            if len(previous_coin_positions):
                # THERE WAS STILL A COIN ON THE FIELD
                reward_sum -= 4
            elif previous_steps_to_next_crate == 0:
                # AGENT WAS NEXT TO CRATE
                reward_sum += 4
            else:
                # AGENT WAS NOT NEXT TO CRATE
                reward_sum -= 5
    # elif len(current_bomb_positions) > 0:
    elif not can_drop_bomb(old_game_state):
        # THERE WAS AND STILL IS A BOMB/EXPLOSION -> OBJECTIVE: DODGE BOMB/EXPLOSIONS BY MOVING TO SAFE FIELD (OR STAYING THERE)
        # if np.min(get_steps_between(current_agent_position, current_bomb_positions)) == 0:
        #     reward_sum -= 10

        ret_safe_fields = find_next_safe_field(previous_map, previous_agent_position)

        if ret_safe_fields is None:
            # SAFE FIELD WAS NOT REACHABLE
            # reward_sum -= 20
            pass
        else:
            # SAFE FIELD WAS REACHABLE
            previous_safe_field_direction, previous_steps_to_secure_field = ret_safe_fields

            if previous_safe_field_direction != 0:
                # AGENT KNEW COIN DIRECTION
                if direction_mapping[previous_safe_field_direction] != action:
                    # AGENT TOOK OTHER ACTION
                    reward_sum -= 4
                else:
                    # AGENT FOLLOWED OUR GUIDANCE
                    reward_sum += 4
            elif action == 'WAIT':
                reward_sum += 4
            else:
                reward_sum -= 4

            # ret = find_next_safe_field(current_map, current_agent_position)

            # if ret is None:
            #     # SAFE FIELD IS NOT REACHABLE
            #     reward_sum -= 25
            # else:
            #     # SAFE FIELD IS REACHABLE
            #     _, current_steps_to_secure_field = ret
            #
            #     if current_steps_to_secure_field < previous_steps_to_secure_field:
            #         # AGENT MOVED CLOSER TO SAFE FIELD
            #         if current_steps_to_secure_field == 0:
            #             # AGENT IS AT SAFE FIELD
            #             reward_sum += 25
            #         else:
            #             # AGENT IS NOT YET AT SAFE FIELD
            #             reward_sum += 20
            #     elif current_steps_to_secure_field > previous_steps_to_secure_field:
            #         # AGENT MOVED AWAY FROM SAFE FIELD
            #         reward_sum -= 20
            #     elif current_steps_to_secure_field != 0:
            #         # AGENT DID NOT MOVE AND IS NOT AT SAFE FIELD
            #         reward_sum -= 15
    elif can_drop_bomb(old_game_state):
        # can_drop_bomb(previous_bomb_positions, previous_explosion_map):
        # THERE WERE NO BOMBS/EXPLOSION -> OBJECTIVE: COLLECT COINS/MOVE CLOSER TO CRATE

        move_to_crate = True

        if len(previous_coin_positions):
            # THERE IS AT LEAST ONE COIN ON THE FIELD -> OBJECTIVE: COLLECT COIN(S)
            ret_coins = find_next_coin(previous_map, previous_agent_position, previous_coin_positions)

            if ret_coins is not None:
                # THERE WAS AT LEAST ONE COIN ON THE FIELD
                move_to_crate = False

                ret_coins = find_next_coin(previous_map, previous_agent_position, previous_coin_positions)

                previous_coin_direction, previous_steps_to_coin = ret_coins

                if previous_coin_direction != 0 and previous_coin_direction != 5:
                    # AGENT KNEW COIN DIRECTION
                    if direction_mapping[previous_coin_direction] != action:
                        # AGENT TOOK OTHER ACTION
                        reward_sum -= 4
                    else:
                        # AGENT FOLLOWED OUR GUIDANCE
                        reward_sum += 4
                else:
                    move_to_crate = True
        if move_to_crate:
            # THERE IS NO COIN/COIN IS NOT REACHABLE -> OBJECTIVE: MOVE TO CRATE
            ret_crates = find_next_crate(previous_map, previous_agent_position, previous_crate_positions)

            if ret_crates is not None:
                # CRATE WAS REACHABLE
                previous_crate_direction, previous_steps_to_crate = ret_crates

                if previous_crate_direction != 0 and previous_crate_direction != 5:
                    # AGENT KNEW CRATE DIRECTION
                    if direction_mapping[previous_crate_direction] != action:
                        # AGENT TOOK OTHER ACTION
                        reward_sum -= 4
                    else:
                        # AGENT FOLLOWED OUR GUIDANCE
                        reward_sum += 4
                elif previous_crate_direction == 0 and can_drop_bomb(old_game_state):
                    # AGENT WAS NEXT TO CRATE AND DID NOT DROP BOMB
                    reward_sum -= 4

                # ret_crates = find_next_crate(current_map, current_agent_position, current_crate_positions)

                # if ret_crates is not None:
                #     # CRATE IS REACHABLE
                #     # ALWAYS TRUE IF THE AGENT DOES NOT DROP A BOMB WHEN TRYING TO MOVE TO THE CRATE
                #     _, current_steps_to_crate = ret_crates
                #
                #     if current_steps_to_crate < previous_steps_to_crate:
                #         # AGENT MOVED CLOSER TO CRATE
                #         reward_sum += 25
                #     elif current_steps_to_crate == previous_steps_to_crate and not can_drop_bomb(previous_bomb_positions, previous_explosion_map) and e.WAITED in events:
                #         # AGENT WAITED NEXT TO CRATE AND COULDN'T DROP A BOMB
                #         reward_sum += 10
                #     else:
                #         # AGENT MOVED AWAY FROM CRATE
                #         reward_sum -= 25

    return reward_sum
