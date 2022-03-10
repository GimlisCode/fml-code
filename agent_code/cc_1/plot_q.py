import pickle

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open("../cc_2/model.pt", "rb") as file:
        Q = pickle.load(file)

    plt.gray()
    plt.imshow(np.reshape(Q, (5, 4)))
    # plt.savefig("q.svg")
    plt.show()

