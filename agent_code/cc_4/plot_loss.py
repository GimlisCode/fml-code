import matplotlib.pyplot as plt
import json


def plot_loss(file="loss.json"):
    with open(file, "r") as f:
        train_info = json.loads(f.read())
        plt.plot(train_info["loss"])
        plt.show()


if __name__ == '__main__':
    plot_loss()
