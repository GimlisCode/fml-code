from typing import List

from torch.utils.data import DataLoader

import events as e
from .callbacks import *

from .dataset import GameStateDataset


def setup_training(self):
    self.Q.learning_rate = 0.001
    self.Q.gamma = 0.25

    self.optimizer = self.Q.configure_optimizers()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    if len(np.array(new_game_state["coins"])) == 0:
        return

    self.train_data["reward"].append(calculate_reward(events, old_game_state, new_game_state))
    self.train_data["state_features_t"].append(state_to_features(old_game_state).tolist())
    self.train_data["state_features_{t+1}"].append(state_to_features(new_game_state).tolist())
    self.train_data["action"].append(get_idx_for_action(self_action))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    dataset = GameStateDataset.from_dict(self.train_data)
    train_loader = DataLoader(dataset, batch_size=40, shuffle=True)

    self.Q.train()

    for i, data in enumerate(train_loader, 0):
        state_features_t, action, reward, state_features_t_plus_1 = data[0].to(self.device), \
                                                                    data[1].to(self.device), \
                                                                    data[2].to(self.device), \
                                                                    data[3].to(self.device)

        if state_features_t.shape[0] == 1:
            # if there is only one element in the last batch
            continue

        self.optimizer.zero_grad()
        loss = self.Q.training_step((state_features_t, action, reward, state_features_t_plus_1), i)
        loss.backward()
        self.optimizer.step()

        self.train_info["train_loss"].append(loss.item())

    self.Q.eval()
    self.Q.save("model.pt")

    if len(self.train_data["action"]) > len(dataset):
        # remove double entries
        # print(f"{len(self.train_data['action'])} > {len(dataset)}")
        del self.train_data
        self.train_data = dataset.to_dict()

    del dataset
    del train_loader

    with open("train_info.json", "w") as file:
        file.write(json.dumps(self.train_info))

    with open("train_data.json", "w") as file:
        file.write(json.dumps(self.train_data))


def calculate_reward(events, old_game_state, new_game_state) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -10

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    previous_min_dist = np.min(get_steps_between(np.array(old_game_state["self"][3]), np.array(old_game_state["coins"])))
    current_min_dist = np.min(get_steps_between(np.array(new_game_state["self"][3]), np.array(new_game_state["coins"])))

    if current_min_dist < previous_min_dist:
        reward_sum += 3
    else:
        reward_sum -= -5

    return np.clip(reward_sum, a_min=-1, a_max=1).item()
