import random
import math
import numpy as np
from tqdm import tqdm
import cv2

from util import read_data, write_data, get_output_dir


# def gonzalez():
#         pass

def multivariate_normal(x, u, v):
    dims = len(x)

    v_det = np.linalg.det(v)
    if v_det == 0:
        raise RuntimeError('Covariance matrix must be non-singular and invertible')

    c = 1 / math.sqrt((2*math.pi)**dims * v_det)
    x_u = np.matrix(np.subtract(x, u))
    v_inv = np.linalg.inv(v)
    return c * math.exp(-0.5 * (x_u * v_inv * x_u.T))


def dist2(a, b):
    return np.linalg.norm(np.subtract(a, b)) ** 2


def diff_outer(a, b):
    c = np.subtract(a, b)
    return np.outer(c, c)


def kmeanspp(X):
    """
    K-means++ initialization
    X is a list of CIE-Lab values: [[L, a, b]]
    """    
    # Choose one pt uniformly at random to be first center
    j = random.randint(0, len(X)-1)
    c1 = X[j]

    # For each pt x, calculate D(x) - distance between x and first center
    W_prefix_sums = []
    W = 0
    for _, x in enumerate(X):
        W += dist2(c1, x)
        W_prefix_sums.append(W)

    # Choose new pt x as second center with probability proportional to D(x)^2
    c2 = None
    q = random.uniform(0, W)
    for i, w in enumerate(W_prefix_sums):
        if w <= q:
            c2 = X[i]
            break
    
    return c1, c2
    

def kmeans_init(X, T):
    """
    X is a list of CIE-Lab values: [[L, a, b]]
    T: number of iterations
    """
    c1, c2 = kmeanspp(X)
    assignments = [None] * len(X)
    
    def assign():
        for i, x in enumerate(X):
            if dist2(x, c1) < dist2(x, c2):
                assignments[i] = 1
            else:
                assignments[i] = 2

    def update():
        nonlocal c1, c2
        c1 = np.zeros_like(c1)
        c2 = np.zeros_like(c2)
        n1 = n2 = 0
        
        for i, x in enumerate(X):
            if assignments[i] == 1:
                n1 += 1
                c1 = np.add(c1, x)
            else:
                n2 += 1
                c2 = np.add(c2, x)
        
        c1 = np.divide(c1, n1)
        c2 = np.divide(c2, n2)

    for _ in tqdm(range(T)):
        assign()
        update()
    
    n1 = n2 = 0
    v1 = np.zeros((3, 3))
    v2 = np.zeros((3, 3))
    for i, x in enumerate(X):
        if assignments[i] == 1:
            v1 = np.add(v1, diff_outer(x, c1))
            n1 += 1
        else:
            v2 = np.add(v2, diff_outer(x, c2))
            n2 += 1
    
    v1 = np.divide(v1, n1)
    v2 = np.divide(v2, n2)

    return (c1, v1, n1), (c2, v2, n2)
        

def em(filename, T, kmeans_init_iters=10, converge_limit=None):
    """
    EM-algorithm
    filename: path to image txt file
    T: max number of iterations
    kmeans_init_iters: number of iterations to run k-means++ for initialization
    converge_limit: stop when the log likelihood differs by less than this amount between iterations
    """

    # data is [[x, y, L, a, b]]
    data, image = read_data(filename, False)
    img_name = filename.split('/')[-1].split('.')[0]

    X = data[:,2:]
    N = len(X)
    d1, d2 = kmeans_init(X, kmeans_init_iters)
    u1, v1, n1 = d1
    u2, v2, n2 = d2
    z1 = n1 / (n1 + n2)

    # Responsibilities
    resps = [0] * N

    def wt_prob(x, u, v, w):
        """
        x: observed variable
        u: mean
        v: covariance matrix
        w: weight
        """
        return w * multivariate_normal(x, u, v)

    def e_step():
        """
        Update responsibilities
        """
        for i, x in enumerate(X):
            r1 = wt_prob(x, u1, v1, z1)
            r2 = wt_prob(x, u2, v2, 1-z1)
            resps[i] = r1 / (r1 + r2)
    
    def m_step():
        """
        Update parameters mu, sigma, alpha
        """
        nonlocal u1, u2, v1, v2, z1
        n1 = 0
        u1 = np.zeros_like(u1)
        u2 = np.zeros_like(u2)
        v1 = np.zeros_like(v1)
        v2 = np.zeros_like(v2)
        # Update means
        for i, x in enumerate(X):
            r = resps[i]
            u1 = np.add(u1, np.multiply(x, r))
            u2 = np.add(u2, np.multiply(x, 1-r))
            n1 += r
        u1 = np.divide(u1, n1)
        u2 = np.divide(u2, N-n1)
        # Update convariance matrices
        for i, x in enumerate(X):
            r = resps[i]
            v1 = np.add(v1, np.multiply(diff_outer(x, u1), r))
            v2 = np.add(v2, np.multiply(diff_outer(x, u2), 1-r))
        v1 = np.divide(v1, n1)
        v2 = np.divide(v2, N-n1)
        # Update mixture weight
        z1 = n1 / N
    

    def log_likelihood():
        """
        Returns p(X|Theta)
        """
        p = 0
        for x in X:
            p += math.log(wt_prob(x, u1, v1, z1) + wt_prob(x, u2, v2, 1-z1))
        return p

    ll = -float('inf')
    t_iterator = tqdm(range(T))
    for _ in t_iterator:
        e_step()
        m_step()
        
        if converge_limit is not None:
            new_ll = log_likelihood()
            if abs(new_ll - ll) < converge_limit:
                t_iterator.close()
                break
            ll = new_ll

    mask = np.zeros_like(image)
    for i in tqdm(range(N)):
        x,d = X[i], data[i]
        if wt_prob(x, u1, v1, z1) > wt_prob(x, u2, v2, 1-z1):
            mask[int(d[1]), int(d[0])] = 255

    cv2.imwrite('../output/{}_mask.jpg'.format(img_name), mask)

    seg1_img = np.zeros_like(image)
    seg1_img[mask==255] = image[mask==255]
    seg1_img = (seg1_img * 255).astype(np.uint8)
    cv2.imwrite('../output/{}_seg1.jpg'.format(img_name), seg1_img)

    seg2_img = np.zeros_like(image)
    seg2_img[mask==0] = image[mask==0]
    seg2_img = (seg2_img * 255).astype(np.uint8)
    cv2.imwrite('../output/{}_seg2.jpg'.format(img_name), seg2_img)


if __name__ == '__main__':
    get_output_dir('../output')

    img_files = [
        '../a2/cow.txt',
        '../a2/fox.txt',
        '../a2/owl.txt',
        '../a2/zebra.txt',
    ]

    for img in img_files:
        em(img, 40, converge_limit=0.001)
    