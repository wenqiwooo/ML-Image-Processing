import math
import numpy as np
from tqdm import tqdm
import cv2

from util import read_data, write_data, get_output_dir

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
    Returns N(x|u, v)
    x: observed value
    u: mean
    v: variance
    """
    return 1 / math.sqrt(2*math.pi*v) * math.exp(-0.5 * (x-u)**2 / v)


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + math.exp(-x))


def tanh(x):
    """
    tanh function
    """
    return 2 * sigmoid(2*x) - 1


def gibbs(filename, J, V, T):
    """
    Gibbs sampling algorithm
    filename: path to image txt file
    J: coupling strength
    V: variance of likelihood distribution
    T: number of samples to draw
    """
    Y = load_data(filename)

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
    
    img_name = filename.split('/')[-1].split('.')[0]
    cv2.imwrite('../output/denoise_gibbs_{}.jpg'.format(img_name), X)


def vi(filename, J, V, T):
    """
    Variational inference algorithm
    filename: path to image txt file
    J: coupling strength
    V: variance
    T: number of iterations
    """
    Y = load_data(filename)
    
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

    img_name = filename.split('/')[-1].split('.')[0]
    cv2.imwrite('../output/denoise_vi_{}.jpg'.format(img_name), X)


if __name__ == '__main__':
    get_output_dir('../output')

    img_files = [
        '../a1/1_noise.txt',
        '../a1/2_noise.txt',
        '../a1/3_noise.txt',
        '../a1/4_noise.txt',
    ]

    for img in img_files:
        gibbs(img, 1, 1, 20)
        vi(img, 1, 1, 20)
