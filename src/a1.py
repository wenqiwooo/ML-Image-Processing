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


def load_data(filename):
    _, Y = read_data(filename, True)
    Y[Y == 0] = -1
    Y[Y == 255] = 1
    return Y


def normal_pdf(x, u, v):
    """
    Returns N(y|x, var)
    """
    return 1 / math.sqrt(2*math.pi*v) * math.exp(-0.5 * (x-u)**2 / v)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return 2 * sigmoid(2*x) - 1


def gibbs(J, V, T):
    """
    J: coupling strength
    V: variance of likelihood distribution
    T: number of samples to draw
    """
    Y = load_data('../a1/4_noise.txt')

    # Initialize X to be same as Y
    X = np.array(Y, copy=True)
    N, M = len(X), len(X[0])

    def unary_potential(y, x):
        return normal_pdf(y, x, V)
    
    def pair_potential(a, b):
        return math.exp(J*a*b)
    
    def calc_prob(i, j):
        p0 = unary_potential(Y[i][j], -1)
        p1 = unary_potential(Y[i][j], 1)
        for di, dj in dirs:
            r = i + di
            c = j + dj
            if 0 <= r < N and 0 <= c < M:
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


def vi(J, V, T):
    """
    Variational inference algorithm
    J: coupling strength
    V: variance
    T: number of iterations
    """
    Y = load_data('../a1/1_noise.txt')
    
    X = np.array(Y, copy=True)
    N, M = len(X), len(X[0])

    def mean_field_influence(i, j):
        res = 0
        for di, dj in dirs:
            r = i + di
            c = j + dj
            if 0 <= r < N and 0 <= c < M:
                res += J * X[r,c]
        return res
    
    def likelihood(y, x):
        return normal_pdf(y, x, V)

    def optimize(i, j):
        m = mean_field_influence(i, j)
        lp = likelihood(Y[i,j], 1)
        lm = likelihood(Y[i,j], -1)
        a = m + 0.5 * (lp - lm)
        X[i,j] = tanh(a)
        
    for _ in tqdm(range(T)):
        for i in range(N):
            for j in range(M):
                optimize(i, j)

    X[X < 0] = 0
    X[X > 0] = 255

    cv2.imshow('', X)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # gibbs(1, 1, 10)
    vi(1, 1, 20)