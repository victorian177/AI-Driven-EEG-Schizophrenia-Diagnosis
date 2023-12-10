import time

import numpy as np
from scipy.spatial.distance import cdist


def sigmoid(x, r):
    assert isinstance(r, tuple), 'When Fx = "Sigmoid", r must be a two-element tuple.'
    y = 1 / (1 + np.exp((x - r[1]) / r[0]))
    return y


def default(x, r):
    assert isinstance(r, tuple), 'When Fx = "Default", r must be a two-element tuple.'
    y = np.exp(-(x ** r[1]) / r[0])
    return y


def modsampen(x, r):
    assert isinstance(r, tuple), 'When Fx = "Modsampen", r must be a two-element tuple.'
    y = 1 / (1 + np.exp((x - r[1]) / r[0]))
    return y


def gudermannian(x, r):
    if r <= 0:
        raise Exception('When Fx = "Gudermannian", r must be a scalar > 0.')
    y = np.arctan(np.tanh(r / x))
    y = y / np.max(y)
    return y


def linear(x, r):
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


class fuzzEntropy:
    def __init__(self, window_size, dissimilarity_index, membership_function="linear"):
        self.m = self.window_size = window_size
        self.r = self.dissimilarity_index = dissimilarity_index
        self.mu = self.membership_function = globals()[membership_function.lower()]

    def __computeFuzzyMatrix(self, x, m):
        if x.ndim == 1:
            N = x.shape[0]
            # x = (x - x.mean())/x.std()
            Xm = np.array([x[i : i + m - 1].tolist() for i in range(0, N - m)])
            dm = cdist(Xm, Xm, "euclidean")
            dm = self.mu(dm, self.r)
            phim = np.sum(dm, axis=1) / (N - m + 1)
            # dm = dm<=self.r
            # phim = np.sum(dm.astype(np.int0),axis=1)/(N-m+1)

        # U0 = np.repeat(np.sum(Xm,axis=1),m-1).reshape(Xm.shape[0],Xm.shape[1])
        # Xm = Xm - np.repeat(np.sum(Xm,axis=1),m-1).reshape(Xm.shape[0],Xm.shape[1])
        # Xm_1 = Xm_1 - np.repeat(np.sum(Xm_1,axis=1),m-2).reshape(Xm_1.shape[0],Xm_1.shape[1])
        # dm = np.zeros((len(Xm),len(Xm)))
        # dm_1 = np.zeros((len(Xm),len(Xm)))
        # for i in range(N-self.d):
        #     for j in range(N-self.d):
        #         if i != j:
        #             dm[i,j]=abs(Xm[i,:]-Xm[j,:]).max()
        #             dm_1[i,j]=abs(Xm_1[i,:]-Xm_1[j,:]).max()
        # Dm = self.mu(self.r).fuzzyValue(dm)
        # Dm_1 = self.mu(self.r).fuzzyValue(dm_1)
        # phim = (1/(N-m+1)) * np.sum(dm,axis=1)
        # phim1 = (1/(N-m-1)) * np.sum(dm_1,axis=1)

        # return np.sum(dm)/(N-self.d)
        return phim

    def _fuzzyEntropyCompute(self, x):
        N = x.shape[0]
        phim = self.__computeFuzzyMatrix(x, self.m)
        phim1 = self.__computeFuzzyMatrix(x, self.m + 1)

        psim = (1 / (N - self.m + 1)) * np.sum(phim, axis=0)
        psim1 = (1 / (N - self.m + 2)) * np.sum(phim1, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            fuzz = np.log(psim) - np.log(psim1)
            # fuzz = np.log(psim)
            # fuzz = psim
        return fuzz


def fuzzEntropy2D(x, window_size, dissimilarity_index, membership_function=linear):
    fuzzyent = fuzzEntropy(window_size, dissimilarity_index, membership_function)

    res = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        res[i] = fuzzyent._fuzzyEntropyCompute(x[i, :])
    return res.mean()


class chatGPTfuzzyEntropy:
    def __init__(self):
        pass

    def _embedding(self, data, m, d):
        N = len(data)
        return np.array([data[i : i + m] for i in range(N - d + 1)])

    def _prob_matrix(self, embedding):
        dists = cdist(embedding, embedding, "euclidean")
        return np.array(
            [np.sum(dists[i] <= dists[i, -1]) - 1 for i in range(len(embedding))]
        ) / float(len(embedding) - 1)

    def fuzzy_entropy(self, data, m, d, r):
        embedding = self._embedding(data, m, d)

        npoints = len(embedding)
        patten_match = np.zeros((npoints, npoints))

        for i in range(npoints):
            for j in range(npoints):
                if (max(abs(embedding[i] - embedding[j]))) <= r:
                    patten_match[i, j] = 1

        p = np.sum(self._prob_matrix(embedding[patten_match == 1])) / float(
            npoints * (npoints - 1)
        )

        return -np.log(p)
