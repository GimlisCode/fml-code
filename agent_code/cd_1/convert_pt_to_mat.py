import pickle

import scipy.io

if __name__ == '__main__':
    with open("model.pt", "rb") as file:
        Q = pickle.load(file)
        scipy.io.savemat("model.mat", mdict={"q":Q})


