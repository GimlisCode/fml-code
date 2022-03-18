from typing import List, Optional
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_data(directory: str, relative: bool = False):
    directory = Path(directory)

    x = [0]
    y = [0]

    for file in list(directory.glob("*.pt")):
        x.append(int(file.name.replace(".pt", "").split("_")[1]))
        with open(file, "rb") as f:
            Q = pickle.load(f)

            if relative:
                y.append(np.average(Q != 0))
            else:
                y.append(np.sum(Q != 0))

    return np.array(x), np.array(y)


def plot_model_population(
        xs: List[np.array],
        ys: List[np.array],
        label: List[str],
        max_lim: bool = True,
        relative: bool = False,
        max_val: Optional[int] = None,
        save_to: Optional[str] = None
):
    if not relative and max_lim and max_val is None:
        raise ValueError("A max value is needed")

    for idx in range(len(xs)):
        x = xs[idx]
        y = ys[idx]

        sorted_experiment_idx = np.argsort(x)
        plt.plot(x[sorted_experiment_idx], y[sorted_experiment_idx], label=label[idx])

    if max_lim:
        plt.ylim(0, 1 if relative else max_val)

    plt.xlabel("Episodes")
    plt.ylabel("Model Population")
    plt.grid(axis="y")
    plt.legend()

    if save_to is not None:
        plt.savefig(save_to)

    plt.show()


if __name__ == '__main__':
    xs = list()
    ys = list()

    experiments = {
        "no improvements": "../../ExperimentData/noAugNoPar/6",
        "augmentations": "../../ExperimentData/augmentation/12",
        "parallel": "../../ExperimentData/parallel(noAug)/combined_snapshots",
        "parallel and augmentation": "../../ExperimentData/parallel+augmentation/combined_snapshots"
    }

    for experiment_key in experiments:
        x, y = load_data(experiments[experiment_key], relative=True)

        xs.append(x)
        ys.append(y)

    plot_model_population(xs, ys, list(experiments.keys()), relative=True,
                          save_to="model_population_different_methods.pdf")
