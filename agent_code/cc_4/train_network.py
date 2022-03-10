import json
from pathlib import Path

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch
import numpy as np

from agent_code.cc_4.dataset import GameStateDataset
from agent_code.cc_4.network import QNetwork


def train(network: QNetwork, data_path, num_of_epochs: int = 25, save_to: str = "model.pt"):
    logging_path = Path("./training_logs")

    if not logging_path.exists():
        logging_path.mkdir()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = GameStateDataset.from_data_path(data_path, device=device)
    print(f"Dataset length: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = network.configure_optimizers()
    schedular = StepLR(optimizer, 15, gamma=0.7)

    network.train()
    network.to(device)

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

        if len(running_loss) == 0 or np.mean(epoch_loss) < np.min(running_loss):
            network.save(str(logging_path.joinpath(f"model_epoch_{epoch+1}.pt")))

        running_loss.append(np.mean(epoch_loss))
        print(f"epoch {epoch + 1}/{num_of_epochs} loss: {running_loss[-1]}")

        schedular.step()

    network.eval()

    network.save(save_to)

    return running_loss


if __name__ == '__main__':
    Q = QNetwork(features_out=4, learning_rate=0.001)
    data_path = "../cc_train_data_collector_1/train_data_new"

    loss = train(Q, data_path, num_of_epochs=50)

    with open("./loss.json", "w") as f:
        f.write(json.dumps({"loss": loss}))
