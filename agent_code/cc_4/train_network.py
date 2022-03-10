import json

from torch.utils.data import DataLoader
import torch
import numpy as np

from agent_code.cc_4.dataset import GameStateDataset
from agent_code.cc_4.network import QNetwork


def train(network: QNetwork, data_path, num_of_epochs: int = 25, save_to: str = "model.pt"):
    dataset = GameStateDataset(data_path)
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = network.configure_optimizers()

    network.train()

    running_loss = list()

    for epoch in range(num_of_epochs):
        epoch_loss = list()

        for i, data in enumerate(train_loader):
            state_features_t, action, reward, state_features_t_plus_1 = data

            if state_features_t.shape[0] == 1:
                # if there is only one element in the last batch
                continue

            optimizer.zero_grad()
            loss = network.training_step((state_features_t, action, reward, state_features_t_plus_1), i)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        running_loss.append(np.mean(epoch_loss))
        print(f"epoch {epoch + 1}/{num_of_epochs} loss: {running_loss[-1]}")

    network.eval()

    network.save(save_to)

    return running_loss


if __name__ == '__main__':
    Q = QNetwork(features_out=4, learning_rate=0.0005)
    data_path = "../cc_train_data_collector_1/train_data_new"

    loss = train(Q, data_path, num_of_epochs=15)

    with open("./loss.json", "w") as f:
        f.write(json.dumps({"loss": loss}))
