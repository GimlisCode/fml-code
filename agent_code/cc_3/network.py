from typing import Optional

import torch
import pytorch_lightning as pl


class QNetwork(pl.LightningModule):
    def __init__(self, features_in: int, features_out: int, gamma: float = 0.5, learning_rate: Optional[float] = 0.0001):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(features_in, out_features=16),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, 6),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(6),
            torch.nn.Linear(6, 4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, features_out),
            torch.nn.ReLU()
        )

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.double()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers.forward(x)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def td_loss(self, y_t, action, reward, y_t_plus_1) -> torch.tensor:
        return torch.mean(torch.square(torch.clip(self.gamma * torch.max(y_t_plus_1, dim=1, keepdim=True)[0] + reward, min=0) - y_t[:, action]))

    def training_step(self, train_batch, batch_idx):
        state_features_t, action, reward, state_features_t_plus_1 = train_batch

        with torch.no_grad():
            y_t = self.forward(state_features_t)
        y_t_plus_1 = self.forward(state_features_t_plus_1)

        loss = self.td_loss(y_t, action, reward, y_t_plus_1)

        # self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        state_features_t, action, reward, state_features_t_plus_1 = val_batch

        y_t = self.forward(state_features_t)
        y_t_plus_1 = self.forward(state_features_t_plus_1)

        loss = self.td_loss(y_t, action, reward, y_t_plus_1)

        # self.log('val_loss', loss)
        return loss

    def load(self, path):
        # self.load_state_dict(torch.load(STATE_DICT_DIR.joinpath("base.ckpt"))["state_dict"])
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
