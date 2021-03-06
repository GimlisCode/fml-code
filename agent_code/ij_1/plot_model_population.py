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
        plt.savefig(save_to, bbox_inches="tight")

    plt.show()


def plot_train_method_populations():
    xs = list()
    ys = list()

    experiments = {
        "no improvements": "../../ExperimentData/Methods/noImprovements/6",
        "AC": "../../ExperimentData/Methods/ac(NoAugNoPar)/3",
        "AUG": "../../ExperimentData/Methods/aug(NoAcNoPar)/2",
        "AUG & AC": "../../ExperimentData/Methods/acAug(noPar)/5",
        "PAR": "../../ExperimentData/Methods/parallel(noAugNoAc)/combined_snapshots",
        "PAR & AC": "../../ExperimentData/Methods/parAc(NoAug)/combined_snapshots",
        "PAR & AUG": "../../ExperimentData/Methods/parAug(NoAc)/combined_snapshots",
        "all": "../../ExperimentData/Methods/parAcAug/combined_snapshots",
    }

    for experiment_key in experiments:
        x, y = load_data(experiments[experiment_key], relative=True)

        xs.append(x)
        ys.append(y)

    plot_model_population(xs, ys, list(experiments.keys()), relative=True,
                          save_to="model_population_different_methods.pdf")


def plot_different_environment_model_populations():
    xs = list()
    ys = list()

    experiments = {
        "baseline": "../../ExperimentData/environments/JustCD/combined_snapshots",
        "trained above": "../../ExperimentData/environments/trainedAbove/combined_snapshots",
        "trained single": "../../ExperimentData/environments/eachTrainedSingle/combined_snapshots",
    }

    for experiment_key in experiments:
        x, y = load_data(experiments[experiment_key], relative=True)

        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(10, 4))
    plot_model_population(xs, ys, list(experiments.keys()), relative=True, max_lim=False,
                          save_to="model_population_different_environments.pdf")


if __name__ == '__main__':
    plot_train_method_populations()
    plot_different_environment_model_populations()
