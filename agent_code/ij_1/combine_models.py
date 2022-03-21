from typing import List, Union
import pickle
from pathlib import Path


def combine(model_files: List[Union[str, Path]], output_file: Union[str, Path]):
    with open(model_files.pop(0), "rb") as file:
        combined_model = pickle.load(file)

    for model in model_files:
        with open(model, "rb") as file:
            q = pickle.load(file)
        combined_model += q

    combined_model /= len(model_files) + 1
    with open(output_file, "wb") as file:
        pickle.dump(combined_model, file)


def combine_parallel_snapshots(snapshot_folder: str, output_path: str):
    snapshot_folder = Path(snapshot_folder)
    output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir()

    parallel_snapshot_folders = [folder for folder in snapshot_folder.iterdir() if folder.is_dir()]

    model_names = [f.name for f in parallel_snapshot_folders[0].glob("*.pt")]

    all_model_names = [f.name for folder in parallel_snapshot_folders for f in folder.glob("*.pt")]
    max_number = max([int(name.replace(".pt", "").split("_")[1]) for name in all_model_names])

    for folder in parallel_snapshot_folders:
        repeat_last_snapshot(folder, max_number)

    for model_name in model_names:
        file_names = [folder.joinpath(model_name) for folder in parallel_snapshot_folders]
        combine(file_names, output_path.joinpath(model_name))


def repeat_last_snapshot(folder: Path, max_number: int):
    model_names = [f.name for f in folder.glob("*.pt")]

    if f"model_{max_number}.pt" in model_names:
        return

    contained_numbers = [int(name.replace(".pt", "").split("_")[1]) for name in model_names]
    last_snapshot_number = max(contained_numbers)
    contained_numbers.remove(last_snapshot_number)
    snapshot_distance = last_snapshot_number - max(contained_numbers)

    print(f"Repeating model {last_snapshot_number} for folder {folder}")

    with open(folder.joinpath(f"model_{last_snapshot_number}.pt"), "rb") as file:
        last_q = pickle.load(file)

    for num in range(last_snapshot_number + snapshot_distance, max_number + snapshot_distance, snapshot_distance):
        with open(folder.joinpath(f"model_{num}.pt"), "wb") as file:
            pickle.dump(last_q, file)


if __name__ == '__main__':
    combine_parallel_snapshots("../../ExperimentData/par(NoArgNoAc)/snapshots",
                               "../../ExperimentData/par(NoArgNoAc)/combined_snapshots")

    combine_parallel_snapshots("../../ExperimentData/parAc(NoArg)/snapshots",
                               "../../ExperimentData/parAc(NoArg)/combined_snapshots")

    combine_parallel_snapshots("../../ExperimentData/parArg(NoAc)/snapshots",
                               "../../ExperimentData/parArg(NoAc)/combined_snapshots")

    combine_parallel_snapshots("../../ExperimentData/parArgAc/snapshots",
                               "../../ExperimentData/parArgAc/combined_snapshots")

    combine_parallel_snapshots("../../ExperimentData/environments/trainedAbove/snapshots",
                               "../../ExperimentData/environments/trainedAbove/combined_snapshots")

    combine_parallel_snapshots("../../ExperimentData/environments/eachTrainedSingle/snapshots",
                               "../../ExperimentData/environments/eachTrainedSingle/combined_snapshots")
