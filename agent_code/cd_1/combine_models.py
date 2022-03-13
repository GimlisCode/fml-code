import pickle
import sys
from os.path import isfile, join
from os import listdir

if __name__ == '__main__':
    path = "../ij_1"
    all_model_files_in_dir = [f for f in listdir(path) if isfile(join(path, f)) and "model" in f and ".pt" in f]

    if len(all_model_files_in_dir) == 0:
        sys.exit()

    with open(path + "/" + all_model_files_in_dir.pop(0), "rb") as file:
        combined_model = pickle.load(file)

    for model in all_model_files_in_dir:
        with open(path + "/" + model, "rb") as file:
            q = pickle.load(file)
        combined_model += q

    combined_model /= len(all_model_files_in_dir)
    with open(f"{path}/model.pt", "wb") as file:
        pickle.dump(combined_model, file)


    # print(all_model_files_in_dir)
    # for file in all_files_in_dir.