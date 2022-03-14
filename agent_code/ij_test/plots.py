import json

import numpy as np
import matplotlib.pyplot as plt


def plot_num_of_negative_rewards(rewards_per_episode: list):
    percentage_of_negative_rewards_per_episode = list()

    for episode in rewards_per_episode:
        percentage_of_negative_rewards_per_episode.append(np.mean(np.array(episode) <= 0))

    plt.boxplot(percentage_of_negative_rewards_per_episode)
    plt.show()


if __name__ == '__main__':
    with open("rewards.json", "r") as f:
        plot_num_of_negative_rewards(json.loads(f.read())["rewards"])
