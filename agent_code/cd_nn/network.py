from typing import Optional

import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class QNetwork(pl.LightningModule):
    def __init__(self, features_in: int, features_out: int, gamma: float = 0.25, learning_rate: Optional[float] = 0.0001):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(features_in, out_features=16),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, 12),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(12),
            torch.nn.Linear(12, 8),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, features_out),
            torch.nn.Softmax(-1)
        )

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.double()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def td_loss(self, y_t, action, reward, y_t_plus_1) -> torch.tensor:
        td_target = self.gamma * torch.max(y_t_plus_1, dim=1, keepdim=True)[0] + reward
        td_target = torch.clip(td_target, min=0)
        return F.smooth_l1_loss(y_t[[range(len(action))], action], td_target)

    def training_step(self, train_batch, batch_idx):
        state_features_t, action, reward, state_features_t_plus_1 = train_batch

        with torch.no_grad():
            y_t_plus_1 = self.forward(state_features_t_plus_1)

        y_t = self.forward(state_features_t)

        loss = self.td_loss(y_t, action, reward, y_t_plus_1)

        if self.trainer is not None:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        state_features_t, action, reward, state_features_t_plus_1 = val_batch

        y_t = self.forward(state_features_t)
        y_t_plus_1 = self.forward(state_features_t_plus_1)

        loss = self.td_loss(y_t, action, reward, y_t_plus_1)

        if self.trainer is not None:
            self.log("val_loss", loss)
        return loss

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def load_from_pl_checkpoint(self, path):
        self.load_state_dict(torch.load(path)["state_dict"])

    def save(self, path):
        torch.save(self.state_dict(), path)
