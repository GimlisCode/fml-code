import pickle

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open("model.pt", "rb") as file:
        Q = pickle.load(file)

    plt.gray()
    plt.imshow(np.reshape(Q, (9 * 27, 4)))
    plt.show()
