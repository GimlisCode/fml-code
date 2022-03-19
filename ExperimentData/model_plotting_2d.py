import pickle
from pathlib import Path


def convert(pt_folder, mat_folder):
    pt_files = Path(pt_folder)
    output_path = Path(mat_folder)

    if not output_path.exists():
        output_path.mkdir()

    pt_file_names = [f for f in pt_files.iterdir() if f.is_file() and "model" in f.name and ".pt" in f.name]
    for model_name in pt_file_names:
        with open(model_name, "rb") as file:
            Q = pickle.load(file)
            scipy.io.savemat(output_path.joinpath(model_name.name.replace(".pt", ".mat")), mdict={"q": Q})


if __name__ == '__main__':
    inputPath = "../../ExperimentData/noAugNoPar/6"
    
    convert(inputPath, outputPath)