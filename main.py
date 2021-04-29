import numpy as np
from time import time, sleep
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


class GameofLife:
    def __init__(self, dim, max_step, seed, sparseness=0.5):

        self.dim = dim  # dimension along one direction
        self.max_step = max_step  # maximum number of timestep

        # must be in (0, 1). 0.5 = equal distribution. the closer to 0, the emptier the system
        self.sparseness = sparseness

        # initiate a random board
        np.random.seed(seed)
        self.board = np.random.choice([0, 1], p=[1-self.sparseness, self.sparseness], size=[self.dim, self.dim])

    def _step_unoptimized(self):

        temp_board = np.zeros((self.dim, self.dim))

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                temp_sum = self.board[(i + 1) % self.dim, j % self.dim] \
                           + self.board[i % self.dim, (j + 1) % self.dim] \
                           + self.board[(i - 1) % self.dim, j % self.dim] \
                           + self.board[i, (j - 1) % self.dim] \
                           + self.board[(i + 1) % self.dim, (j + 1) % self.dim] \
                           + self.board[(i - 1) % self.dim, (j + 1) % self.dim] \
                           + self.board[(i - 1) % self.dim, (j - 1) % self.dim] \
                           + self.board[(i + 1) % self.dim, (j - 1) % self.dim]

                if self.board[i, j] == 1:
                    if temp_sum < 2 or temp_sum > 3:
                        temp_board[i, j] = 0
                    else:
                        temp_board[i, j] = 1

                elif self.board[i, j] == 0:
                    if temp_sum == 3:
                        temp_board[i, j] = 1

        self.board = temp_board

    def _step_optimized(self):  # calculate each step
        conv_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        temp_board = convolve2d(self.board, conv_kernel, mode="same", boundary="wrap")

        self.board = np.logical_or(np.logical_and((self.board == 1), np.logical_or(temp_board == 2, temp_board == 3)),
                                   np.logical_and((self.board == 0), (temp_board == 3)))
        
    def run_and_plot(self, optimized=True, pause_between_frames=0.05):
        fig, ax = plt.subplots()
        ln = ax.imshow(self.board, animated=True, cmap="binary")
        plt.show(block=False)
        plt.pause(0.1)
        bg = fig.canvas.copy_from_bbox(fig.bbox)
        ax.draw_artist(ln)
        fig.canvas.blit(fig.bbox)

        for j in range(self.max_step):
            if optimized:
                self._step_optimized()
            else:
                self._mixed()

            fig.canvas.restore_region(bg)
            ln = ax.imshow(self.board, animated=True, cmap="binary")
            ax.draw_artist(ln)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            sleep(pause_between_frames)


if __name__ == "__main__":
    t1 = time()
    game = GameofLife(dim=150, max_step=25, sparseness=0.5, seed=42)
    game.run_and_plot(optimized=True, pause_between_frames=0.0)
    print("Evaluation time: " + str(time() - t1))
    t2 = time()
    game2 = GameofLife(dim=150, max_step=25, sparseness=0.5, seed=42)
    game2.run_and_plot(optimized=False, pause_between_frames=0.0)

    print(np.sum(np.sum(game.board-game2.board)))
    print("Evaluation time: " + str(time() - t2))
