import time

import numpy as np
from scipy.spatial.distance import cdist

def sigmoid(x, r):
    """
    Sigmoid function.

    Args:
        x (numpy.ndarray): Input data.
        r (tuple): Tuple containing two elements.

    Returns:
        numpy.ndarray: Output of the sigmoid function.
    """
    assert isinstance(r, tuple), 'When Fx = "Sigmoid", r must be a two-element tuple.'
    y = 1 / (1 + np.exp((x - r[1]) / r[0]))
    return y

def default(x, r):
    """
    Default function.

    Args:
        x (numpy.ndarray): Input data.
        r (tuple): Tuple containing two elements.

    Returns:
        numpy.ndarray: Output of the default function.
    """
    assert isinstance(r, tuple), 'When Fx = "Default", r must be a two-element tuple.'
    y = np.exp(-(x ** r[1]) / r[0])
    return y

def modsampen(x, r):
    """
    Modsampen function.

    Args:
        x (numpy.ndarray): Input data.
        r (tuple): Tuple containing two elements.

    Returns:
        numpy.ndarray: Output of the modsampen function.
    """
    assert isinstance(r, tuple), 'When Fx = "Modsampen", r must be a two-element tuple.'
    y = 1 / (1 + np.exp((x - r[1]) / r[0]))
    return y

def gudermannian(x, r):
    """
    Gudermannian function.

    Args:
        x (numpy.ndarray): Input data.
        r (float): Scalar value > 0.

    Returns:
        numpy.ndarray: Output of the gudermannian function.
    """
    if r <= 0:
        raise Exception('When Fx = "Gudermannian", r must be a scalar > 0.')
    y = np.arctan(np.tanh(r / x))
    y = y / np.max(y)
    return y

def linear(x, r):
    """
    Linear function.

    Args:
        x (numpy.ndarray): Input data.
        r (int): 0 or 1.

    Returns:
        numpy.ndarray: Output of the linear function.
    """
    if r == 0 and x.shape[0] > 1:
        y = np.exp(-(x - np.min(x)) / np.ptp(x))
    elif r == 1:
        y = np.exp(-(x - np.min(x)))
    elif r == 0 and x.shape[0] == 1:
        y = 0
    else:
        print(r)
        raise Exception('When Fx = "Linear", r must be 0 or 1')
    return y

class FuzzEntropy:
    """
    Fuzzy entropy class.
    """

    def __init__(self, window_size, dissimilarity_index, membership_function="linear"):
        """
        Constructor for FuzzEntropy.

        Args:
            window_size (int): Size of the window.
            dissimilarity_index (float): Dissimilarity index.
            membership_function (str): Membership function name.
        """
        self.m = self.window_size = window_size
        self.r = self.dissimilarity_index = dissimilarity_index
        self.mu = self.membership_function = globals()[membership_function.lower()]

    def __compute_fuzzy_matrix(self, x, m):
        """
        Compute fuzzy matrix.

        Args:
            x (numpy.ndarray): Input data.
            m (int): Size of the window.

        Returns:
            numpy.ndarray: Fuzzy matrix.
        """
        if x.ndim == 1:
            N = x.shape[0]
            Xm = np.array([x[i : i + m - 1].tolist() for i in range(0, N - m)])
            dm = cdist(Xm, Xm, "euclidean")
            dm = self.mu(dm, self.r)
            phim = np.sum(dm, axis=1) / (N - m + 1)

        return phim

    def _fuzzy_entropy_compute(self, x):
        """
        Compute fuzzy entropy.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Fuzzy entropy.
        """
        N = x.shape[0]
        phim = self.__compute_fuzzy_matrix(x, self.m)
        phim1 = self.__compute_fuzzy_matrix(x, self.m + 1)

        psim = (1 / (N - self.m + 1)) * np.sum(phim, axis=0)
        psim1 = (1 / (N - self.m + 2)) * np.sum(phim1, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            fuzz = np.log(psim) - np.log(psim1)
        return fuzz

def fuzz_entropy_2d(x, window_size, dissimilarity_index, membership_function=linear):
    """
    Compute fuzzy entropy for 2D data.

    Args:
        x (numpy.ndarray): Input data.
        window_size (int): Size of the window.
        dissimilarity_index (float): Dissimilarity index.
        membership_function (function): Membership function.

    Returns:
        float: Fuzzy entropy.
    """
    fuzzy_ent = FuzzEntropy(window_size, dissimilarity_index, membership_function)

    res = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        res[i] = fuzzy_ent._fuzzy_entropy_compute(x[i, :])
    return res.mean()

class ChatGPTFuzzyEntropy:
    """
    Fuzzy entropy for ChatGPT data.
    """

    def __init__(self):
        """
        Constructor for ChatGPTFuzzyEntropy.
        """
        pass

    def _embedding(self, data, m, d):
        """
        Embedding function.

        Args:
            data (numpy.ndarray): Input data.
            m (int): Size of the window.
            d (int): Embedding dimension.

        Returns:
            numpy.ndarray: Embedded data.
        """
        N = len(data)
        return np.array([data[i : i + m] for i in range(N - d + 1)])

    def _prob_matrix(self, embedding):
        """
        Compute probability matrix.

        Args:
            embedding (numpy.ndarray): Embedded data.

        Returns:
            numpy.ndarray: Probability matrix.
        """
        dists = cdist(embedding, embedding, "euclidean")
        return np.array(
            [np.sum(dists[i] <= dists[i, -1]) - 1 for i in range(len(embedding))]
        ) / float(len(embedding) - 1)

    def fuzzy_entropy(self, data, m, d, r):
        """
        Compute fuzzy entropy for ChatGPT data.

        Args:
            data (numpy.ndarray): Input data.
            m (int): Size of the window.
            d (int): Embedding dimension.
            r (float): Dissimilarity index.

        Returns:
            float: Fuzzy entropy.
        """
        embedding = self._embedding(data, m, d)

        npoints = len(embedding)
        patten_match = np.zeros((npoints, npoints))

        for i in range(npoints):
            for j in range(npoints):
                if max(abs(embedding[i] - embedding[j])) <= r:
                    patten_match[i, j] = 1

        p = np.sum(self._prob_matrix(embedding[patten_match == 1])) / float(
            npoints * (npoints - 1)
        )

        return -np.log(p)
