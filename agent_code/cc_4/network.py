from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class QNetwork(pl.LightningModule):
    def __init__(self, features_out: int, gamma: float = 0.5, learning_rate: Optional[float] = 0.0001):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            # expected input 18x18
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 18x18
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 18x18
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 9x9
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 9x9
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(32 * 9 * 9, 264),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(264),
            torch.nn.Linear(264, features_out),
            torch.nn.Softmax(-1)
            # torch.nn.ReLU()
        )

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.double()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linear_layers.forward(self.conv_layers.forward(x).view((-1, 32 * 9 * 9)))

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001, eps=0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def td_loss(self, y_t, action, reward, y_t_plus_1) -> torch.tensor:
        td_target = self.gamma * torch.max(y_t_plus_1, dim=1, keepdim=True)[0] + reward
        td_target = torch.clip(td_target, min=0, max=1)
        return F.smooth_l1_loss(y_t[:, action], td_target)

    def training_step(self, train_batch, batch_idx):
        state_features_t, action, reward, state_features_t_plus_1 = train_batch

        with torch.no_grad():
            y_t_plus_1 = self.forward(state_features_t_plus_1)

        y_t = self.forward(state_features_t)

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
