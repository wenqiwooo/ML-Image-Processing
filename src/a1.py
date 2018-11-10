import math
import numpy as np
from tqdm import tqdm
import cv2

from io_data import read_data, write_data

dirs = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
)

def gibbs(J, V, T):
    """
    J: coupling strength
    V: variance of likelihood distribution
    T: number of samples to draw
    """
    _, Y = read_data('../a1/4_noise.txt', True)
    Y[Y == 0] = -1
    Y[Y == 255] = 1

    # Initialize X to be same as Y
    X = np.array(Y, copy=True)
    # X is a N x M matrix
    N = len(X)
    M = len(X[0])

    def unary_potential(y, x):
        """
        Returns N(y|x, var)
        """
        return 1 / math.sqrt(2*math.pi*V) * math.exp(-0.5 * (y-x)**2 / V)
    
    def pair_potential(a, b):
        return math.exp(J*a*b)
    
    def calc_prob(i, j):
        p0 = unary_potential(Y[i][j], -1)
        p1 = unary_potential(Y[i][j], 1)
        for di, dj in dirs:
            r = i + di
            c = j + dj
            if 0 <= r < M and 0 <= c < N:
                p0 *= pair_potential(-1, X[r][c])
                p1 *= pair_potential(1, X[r][c])
        return p1 / (p0 + p1)

    for _ in tqdm(range(T)):
        for i in range(N):
            for j in range(M):
                p1 = calc_prob(i, j)
                if np.random.uniform() < p1:
                    X[i][j] = 1
                else:
                    X[i][j] = -1
    
    X[X == -1] = 0
    X[X == 1] = 255
    
    cv2.imshow('', X)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gibbs(1, 1, 10)