from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_model_population(directory: str, max_lim: bool = False, relative: bool = False):
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

    x = np.array(x)
    y = np.array(y)

    idx = np.argsort(x)
    plt.plot(x[idx], y[idx])
    if max_lim:
        plt.ylim(0, 1 if relative else np.max(Q))
    plt.show()


if __name__ == '__main__':
    plot_model_population("snapshots", max_lim=False, relative=True)
    plot_model_population("snapshots", max_lim=True, relative=True)
