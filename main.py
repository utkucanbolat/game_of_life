import numpy as np
from time import time
from os import system
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


class GameofLife:
    def __init__(self, dim, max_step, seed, sparseness=0.5):

        self.N = dim  # dimension along one direction
        self.max_step = max_step  # maximum number of timestep

        # must be in (0, 1). 0.5 = equal distribution. the closer to 0, the emptier the system
        self.sparseness = sparseness

        # initiate a random board
        np.random.seed(seed)
        self.board = np.random.choice([0, 1], p=[1-self.sparseness, self.sparseness], size=[self.N, self.N])

    def run(self):
        for frame in range(self.max_step):
            self._plot_and_save_board(frame)
            self._step()
            print("Timestep: " + str(frame), end="\r")

    def _step(self):  # calculate each step
        conv_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        temp_board = convolve2d(self.board, conv_kernel, mode="same", boundary="wrap")

        self.board = np.logical_or(np.logical_and((self.board == 1), np.logical_or(temp_board == 3, temp_board == 2)),
                                   np.logical_and((self.board == 0), (temp_board == 3)))

    def _plot_and_save_board(self, frame):
        plt.ioff()  # to disable figure pop-up
        plt.matshow(self.board, cmap="binary")  # monochromatically plot matrix
        plt.title("Timestep: " + str(frame))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig("frames/frame" + str(frame) + ".png", dpi=600)  # save the figure
        plt.close('all')

    @staticmethod
    def make_movie():  # use ffmpeg to create movie out of the images saved
        system("ffmpeg -r 10 -i frames/frame%01d.png -vcodec mpeg4 -y my_movie.mp4")


if __name__ == "__main__":
    t1 = time()
    game = GameofLife(seed=42, dim=500, max_step=1000, sparseness=0.2)
    game.run()
    print("Evaluation time: " + str(time()-t1))
    game.make_movie()
