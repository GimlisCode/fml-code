import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from agent_code.cc_4.dataset import GameStateDataset
from agent_code.cc_4.network import QNetwork

if __name__ == "__main__":
    all_data = GameStateDataset.from_data_path("../cc_train_data_collector_1/train_data_new", load=False)
    dataset_train, dataset_val = all_data.split(val_percentage=0.2)

    print("dataset length (train/val): ", len(dataset_train), len(dataset_val))

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=16)

    Q = QNetwork(features_out=4, learning_rate=0.001)

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=0.001, patience=25, mode="min"),
        ModelCheckpoint(monitor="val_loss", mode="min")
    ]

    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else 0, max_epochs=25, callbacks=callbacks)
    trainer.fit(Q, train_loader, val_loader)
    trainer.save_checkpoint("./final_epoch.ckpt")
