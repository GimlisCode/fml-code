import itertools
import json
from typing import List, Dict, Any, Union

import matplotlib.pyplot as plt
import numpy as np


def load_agent_performance(path: Union[str, List[str]]):
    def _load_agent_performance(file: str) -> Dict[str, Any]:
        with open(file, "r") as f:
            return json.loads(f.read())

    if isinstance(path, str):
        # load single performance file
        return _load_agent_performance(path)
    elif isinstance(path, list):
        # load multiple given performance files (from parallel test runs)
        performance = _load_agent_performance(path.pop(0))

        for file_path in path:
            next_performance = _load_agent_performance(file_path)

            for key in performance.keys():
                performance[key] += next_performance[key]

        return performance
    else:
        raise NotImplementedError(f"load_agent_performance not implemented for values of type {type(path)}")


def plot_num_of_negative_rewards(rewards_per_episode: List[List[int]], labels: List[str], save_to: str = None):
    data = list()

    for experiment_idx, current_rewards_per_episode in enumerate(rewards_per_episode):
        percentage_of_negative_rewards_per_episode = list()

        for episode in current_rewards_per_episode:
            percentage_of_negative_rewards_per_episode.append(np.mean(np.array(episode) < 0))

        data.append(percentage_of_negative_rewards_per_episode)

    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=12)
    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
    plt.show()


def plot_points(points_per_epoch: List[List[int]], labels: List[str], save_to: str = None):
    data = list()
    for experiment_idx, current_points_per_episode in enumerate(points_per_epoch):
        data.append(current_points_per_episode)

    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=12)
    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
    plt.show()


def to_latex_table(
        experiment_performances: List[Dict[str, List[Any]]],
        names: List[str],
        caption: str,
        label: str,
        out_file: str
):
    output_stream = open(out_file, "w")

    print("\\begin{table}", file=output_stream)
    print("\t\\centering", file=output_stream)
    print(f"\t\\caption{{{caption}}}", file=output_stream)
    print("\t\\begin{tabular}{c|c|c|c|c|c}", file=output_stream)
    print(f"\t\t\\label{{{label}}}", file=output_stream)
    print("\t\t\\multirow{2}*{\\textbf{Experiment}} & \\multicolumn{2}{c|}{\\textbf{Rewards}} & "
          "\\multicolumn{2}{c|}{\\textbf{Points}} & \\multirow{2}*{\\textbf{Deaths (\%)}} \\\\ ",
          file=output_stream)

    print("& \\textbf{Median} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Mean} & \\\\", file=output_stream)

    print("\t\t\\hline", file=output_stream)

    for experiment_idx,  experiment_performance in enumerate(experiment_performances):
        median_reward = round(np.median(list(itertools.chain(*experiment_performance["rewards"]))).item(), 2)
        median_points = round(np.median(experiment_performance["points"]).item(), 2)
        mean_reward = round(np.mean(list(itertools.chain(*experiment_performance["rewards"]))).item(), 2)
        mean_points = round(np.mean(experiment_performance["points"]).item(), 2)
        death_percentage = round(np.mean(experiment_performance["agent_died"]) * 100, 2)

        current_name = names[experiment_idx].replace("&", "\\&")
        print(f"\t\t{current_name} & {median_reward} & {mean_reward} & {median_points} & {mean_points} "
              f"& {death_percentage} \\\\",
              file=output_stream)

    print("\t\t\\hline", file=output_stream)
    print("\t\\end{tabular}", file=output_stream)
    print("\\end{table}", file=output_stream)

    output_stream.close()


def plot_method_performances():
    experiments = {
        "base": "../../ExperimentData/noImprovements/performance.json",
        "AC": "../../ExperimentData/Ac(NoArgNoPar)/performance.json",
        "AUG": "../../ExperimentData/arg(noParNoAc)/performance.json",
        "AUG & AC": "../../ExperimentData/argAc(noPar)/performance.json",
        "PAR": "../../ExperimentData/par(NoArgNoAc)/performance.json",
        "PAR & AC": "../../ExperimentData/parAc(NoArg)/performance.json",
        "PAR & AUG": "../../ExperimentData/parArg(NoAc)/performance.json",
        "all": "../../ExperimentData/parArgAc/performance.json",
    }

    experiment_performances = list()

    for experiment_key in experiments:
        experiment_performances.append(load_agent_performance(experiments[experiment_key]))

    plot_num_of_negative_rewards([performance["rewards"] for performance in experiment_performances],
                                 list(experiments.keys()), save_to="negative_rewards_box_plots.pdf")

    plot_points([performance["points"] for performance in experiment_performances], list(experiments.keys()),
                save_to="points_box_plots.pdf")

    to_latex_table(experiment_performances,
                   list(experiments.keys()),
                   "Comparison of the performance playing against rule based agents.",
                   "tab:experiments_performance_comparison",
                   "comparison_test_run.tex")


def plot_environment_performances():
    experiments = {
        "only classic": [
            "../../ExperimentData/environments/justCD/performance_500_1.json",
            "../../ExperimentData/environments/justCD/performance_500_2.json",
            "../../ExperimentData/environments/justCD/performance_500_3.json",
        ],
        "trained single": [
            "../../ExperimentData/environments/eachTrainedSingle/performance_0.json",
            "../../ExperimentData/environments/eachTrainedSingle/performance_1.json",
            "../../ExperimentData/environments/eachTrainedSingle/performance_2.json",
        ],
        "trained above": [
            "../../ExperimentData/environments/trainedAbove/performance_0.json",
            "../../ExperimentData/environments/trainedAbove/performance_1.json",
            "../../ExperimentData/environments/trainedAbove/performance_2.json",
        ],
    }

    experiment_performances = list()

    for experiment_key in experiments:
        experiment_performances.append(load_agent_performance(experiments[experiment_key]))

    plot_num_of_negative_rewards([performance["rewards"] for performance in experiment_performances],
                                 list(experiments.keys()), save_to="negative_rewards_box_plots_environments.pdf")

    plot_points([performance["points"] for performance in experiment_performances], list(experiments.keys()),
                save_to="points_box_plots_environments.pdf")

    to_latex_table(experiment_performances,
                   list(experiments.keys()),
                   "Comparison of the performance of the model trained on different environments "
                   "playing against rule based agents.",
                   "tab:experiments_performance_comparison_environments",
                   "comparison_test_run_environments.tex")


if __name__ == '__main__':
    plot_method_performances()
    plot_environment_performances()
