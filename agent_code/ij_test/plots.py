import json
from typing import List, Dict, Any
import itertools

import numpy as np
import matplotlib.pyplot as plt


def load_agent_performance(path: str):
    with open(path, "r") as f:
        return json.loads(f.read())


def plot_num_of_negative_rewards(rewards_per_episode: List[List[int]], labels: List[str]):
    data = list()

    for experiment_idx, current_rewards_per_episode in enumerate(rewards_per_episode):
        percentage_of_negative_rewards_per_episode = list()

        for episode in current_rewards_per_episode:
            percentage_of_negative_rewards_per_episode.append(np.mean(np.array(episode) < 0))

        data.append(percentage_of_negative_rewards_per_episode)

    plt.boxplot(data, labels=labels)
    plt.show()


def plot_points(points_per_epoch: List[List[int]], labels: List[str]):
    data = list()
    for experiment_idx, current_points_per_episode in enumerate(points_per_epoch):
        data.append(current_points_per_episode)

    plt.boxplot(data, labels=labels)
    plt.show()


def to_latex_table(
        experiment_performances: List[Dict[str, List[Any]]],
        names: List[str],
        caption: str,
        label: str,
        out_file: str
):
    output_stream = open(out_file, "w")

    """
    \begin{table}
        \centering
        \caption{Comparison of different agent implementations. The points were measured during playing the \ac{ch} scenario and the classic Bomberman scenario versus one coin\_collector enemy and averaged over five rounds. To avoid. In order to avoid long numbers, these have been rounded off. Just the latest versions of the different implementations are considered. Deep Reinforcement Learning implementations are not considered in this overview. Points \ac{ch}: Amount of collected coins (our agent:enemy agent); Points classic: amount of points (our agent:enemy agent); \# training episodes: number of episodes till the model is fully trained for both scenarios. }
        \begin{tabular}{c|l|l|l}
            \label{tab:OverviewTraining}
            \textbf{Implementation} & \textbf{Points CH} & \textbf{Points classic} & \textbf{\# training episodes} \\ 
            \hline
            Coin-Collector 1 & 25:25 & - & ca. 1100 \\ 
            Coin-Collector 2 & 25:25 & - & ca. 200 \\
            Crate-Destroyer 1 & 25:25 & 5:4 & ca. 1500 \\
            Crate-Destroyer 2 & 25:25 & 5:4 & ca. 2200 \\
            Crate-Destroyer 3 & 25:25 & 5:4 & ca. 2200 \\
            Indiana-Jones 1 & 25:25 & 7:4 & ca. 10$\times$2000 \\
            \hline
        \end{tabular}
    \end{table}
    """

    print("\\begin{table}", file=output_stream)
    print("\t\\centering", file=output_stream)
    print(f"\t\\caption{{{caption}}}", file=output_stream)
    print("\t\\begin{tabular}{c|l|l|l}", file=output_stream)
    print(f"\t\t\\label{{{label}}}", file=output_stream)
    print("\t\t\\textbf{Experiment} & \\textbf{Median Reward} & \\textbf{Median Points} & \\textbf{Deaths (\%)} \\\\ ",
          file=output_stream)
    print("\t\t\\hline", file=output_stream)

    for experiment_idx,  experiment_performance in enumerate(experiment_performances):
        mean_reward = round(np.median(list(itertools.chain(*experiment_performance["rewards"]))).item(), 2)
        mean_points = round(np.median(experiment_performance["points"]).item(), 2)
        death_percentage = round(np.mean(experiment_performance["agent_died"]) * 100, 2)

        print(f"\t\t{names[experiment_idx]} & {mean_reward} & {mean_points} & {death_percentage} \\\\",
              file=output_stream)

    print("\t\t\\hline", file=output_stream)
    print("\t\\end{tabular}", file=output_stream)
    print("\\end{table}", file=output_stream)

    output_stream.close()


if __name__ == '__main__':
    experiments = {
        "no improvements": "../../ExperimentData/noAugNoPar/performance.json",
        "augmentations": "../../ExperimentData/augmentation/performance.json",
        "parallel": "../../ExperimentData/parallel(noAug)/performance.json",
        "parallel and augmentation": "../../ExperimentData/parallel+augmentation/performance.json"
    }

    experiment_performances = list()

    for experiment_key in experiments:
        experiment_performances.append(load_agent_performance(experiments[experiment_key]))

    plot_num_of_negative_rewards([performance["rewards"] for performance in experiment_performances],
                                 list(experiments.keys()))

    plot_points([performance["points"] for performance in experiment_performances], list(experiments.keys()))

    to_latex_table(experiment_performances, list(experiments.keys()), "Test", "tab:experiments_comparison", "test.tex")
