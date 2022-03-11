import pickle
import sys
from os.path import isfile, join
from os import listdir

if __name__ == '__main__':
    path = "./"
    all_model_files_in_dir = [f for f in listdir(path) if isfile(join(path, f)) and "model" in f and ".pt" in f]

    if len(all_model_files_in_dir) == 0:
        sys.exit()

    with open(all_model_files_in_dir[0], "rb") as file:
        combined_model = pickle.load(file)

    for model in all_model_files_in_dir:
        with open(model, "rb") as file:
            q = pickle.load(file)
        combined_model += q
        combined_model /= 2 # instant avg to avoid overflow

    with open(f"model.pt", "wb") as file:
        pickle.dump(q, file)


    # print(all_model_files_in_dir)
    # for file in all_files_in_dir.