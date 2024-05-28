import numpy as np
import matplotlib.pyplot as plt


def main():
    losses = np.load("losses2.npy")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
