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

    for model_name in model_names:
        file_names = [folder.joinpath(model_name) for folder in parallel_snapshot_folders]
        combine(file_names, output_path.joinpath(model_name))


if __name__ == '__main__':
    combine_parallel_snapshots("../../ExperimentData/parallel(noAug)/snapshots",
                               "../../ExperimentData/parallel(noAug)/combined_snapshots")

    combine_parallel_snapshots("../../ExperimentData/parallel+augmentation/snapshots",
                               "../../ExperimentData/parallel+augmentation/combined_snapshots")
