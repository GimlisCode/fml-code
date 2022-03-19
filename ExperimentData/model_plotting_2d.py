from typing import List, Optional
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_data(directory: str, snapshot_id, relative: bool = False):
    directory = Path(directory)
    with open(directory.joinpath("model_" + str(snapshot_id) + ".pt"), "rb") as f:
        model = pickle.load(f)

    if relative:
        model = model != 0
        model = model.astype(int)

    return model


def plot_model_2d(
        models,
        title,
        path: Optional[str] = None,
        save_to: Optional[str] = None
):
    fig, axis = plt.subplots(len(models.keys()))
    fig.set_size_inches(17,17)
    # fig.suptitle(title)
    for idx, key in enumerate(models.keys()):
        current_ax = axis[idx]
        current_model = np.array(models[key]).reshape(-1,6).T
        im = current_ax.imshow(current_model, cmap="gray")
        current_ax.set_title(f"New visited states after {key} episodes")
        current_ax.set(ylabel="Action ids", xlabel="State ids")
        current_ax.set_aspect("auto")
        current_ax.label_outer()
        for item in ([current_ax.title, current_ax.xaxis.label, current_ax.yaxis.label] +
                     current_ax.get_xticklabels() + current_ax.get_yticklabels()):
            item.set_fontsize(20)
        fig.colorbar(im, ax=current_ax)

    if save_to is not None:
        if path is not None:
            path = Path(path)
            save_to = path.joinpath(save_to)
        fig.savefig(save_to)

    plt.show()

def plot_model_2d_difference(
        models,
        title,
        path: Optional[str] = None,
        save_to: Optional[str] = None
):
    fig, axis = plt.subplots(len(models.keys())-1)
    fig.set_size_inches(17, 17)
    sorted_keys = sorted(list(models.keys()))
    # fig.suptitle(title)
    for idx in range(1, len(models.keys())):
        key = sorted_keys[idx]
        current_ax = axis[idx-1]
        previous_model = models[sorted_keys[idx-1]]
        current_model = (np.array(models[key]) - np.array(previous_model)).reshape(-1, 6).T
        im = current_ax.imshow(current_model, cmap="gray")
        current_ax.set_title(f"New visited states after {key} episodes")
        current_ax.set(ylabel="Action ids", xlabel="State ids")
        current_ax.set_aspect("auto")
        current_ax.label_outer()
        for item in ([current_ax.title, current_ax.xaxis.label, current_ax.yaxis.label] +
                     current_ax.get_xticklabels() + current_ax.get_yticklabels()):
            item.set_fontsize(20)
        fig.colorbar(im, ax=current_ax)

    if save_to is not None:
        if path is not None:
            path = Path(path)
            save_to = path.joinpath(save_to)
        fig.savefig(save_to)

    plt.show()


if __name__ == '__main__':
    difference = True
    path = "../../ExperimentData/noAugNoPar/6"

    models = dict()
    snapshots = [100, 250, 500, 1000, 1500]

    for snapshotId in snapshots:
        x = load_data(path, snapshotId, relative=True)
        models[snapshotId] = x

    if difference:
        models[0] = np.zeros_like(list(models.keys())[-1])
        plot_model_2d_difference(models, "Learning history for a big model", path, save_to="big_model_learning.pdf")
    else:
        plot_model_2d(models, "Learning history for a big model", path, save_to="improved_model_learning.pdf")