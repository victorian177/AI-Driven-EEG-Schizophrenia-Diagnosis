import numpy as np
import mne
import math
import EntropyHub
from generisAPI.fuzzEntropy import *


def xtract_phase_data(phase, data):
    res = dict()
    for s in data:
        res[s] = dict()
        res[s]["eeg_data"] = data[s]["eeg_data"][phase]
        res[s]["eeg_markers"] = data[s]["eeg_markers"][phase]
    return res


class trial_epoching:
    def __init__(self, mode="min"):
        self.mode = mode
        pass

    def fit_transform(self, X):
        if (X.shape[1] < 200) or (X.shape[1] % 2 != 0):
            print("beta", X.shape)
            return X
        else:
            no_epochs = []
            for i in range(15, 1, -1):
                if X.shape[1] % i == 0:
                    no_epochs.append(i)
            if self.mode == "min":
                x = X.reshape(
                    min(no_epochs), X.shape[0], int(X.shape[1] / min(no_epochs))
                )
            else:
                x = X.reshape(
                    max(no_epochs), X.shape[0], int(X.shape[1] / max(no_epochs))
                )
            return x


class trial_averaging:
    def __init__(self, axis=0, dummy=0):
        self.axis = axis

    def fit_transform(self, X):
        return np.average(X, self.axis)


class trials_averaging:
    def __init__(self):
        pass

    def fit_transform(self, X):
        sum = 0
        for x in X:
            sum += np.array(x)
        sum /= len(X)
        return sum


class stft:
    def __init__(self, fs, sfreq):
        self.fs = fs
        self.w = (fs - 1) * 2
        self.f = mne.time_frequency.stftfreq(self.w, sfreq)

    def __stft(self, x):
        return mne.time_frequency.stft(x, self.w)

    def fit_transform(self, X):
        if X.ndim == 2:
            return self.__stft(X)
        elif X.ndim == 3:
            res = np.empty(
                (X.shape[0], X.shape[1], self.fs, math.ceil(X.shape[2] * 2 / self.w))
            )
            for i in range(X.shape[0]):
                res[i, :, :, :] = self.__stft(X[i, :, :])
        return res


class fuzzEnt:
    def __init__(self, m, r=1, mode="self", muFunction="linear"):
        self.m = m
        self.r = r
        self.mu = muFunction
        self.mode = mode

    def fit_transform(self, X):
        if X.ndim == 3:
            res = np.empty((1))
            for e in X:
                if e.shape[-1] > 126:
                    for i in range(30, 1, -1):
                        if e.shape[-1] % i == 0 and e.shape[-1] / i <= 126:
                            break
                    cnt = int(e.shape[-1] / i)
                    if self.mode == "hub":
                        ent = EntropyHub.FuzzEn2D(e[:, 0:cnt], self.m)
                    else:
                        ent = fuzzEntropy2D(e[:, 0:cnt], self.m, self.r, self.mu)
                    for c in range(1, i):
                        if self.mode == "hub":
                            ent += EntropyHub.FuzzEn2D(
                                e[:, c * cnt : (c + 1) * cnt], self.m
                            )
                        else:
                            ent += fuzzEntropy2D(
                                e[:, c * cnt : (c + 1) * cnt], self.m, self.r, self.mu
                            )
                else:
                    if self.mode == "hub":
                        ent = EntropyHub.FuzzEn2D(e, self.m, Lock=False)
                    else:
                        ent += fuzzEntropy2D(e, self.m, self.r, self.mu)
                res = np.append(res, ent)
            return res
        elif X.ndim == 2:
            if self.mode == "hub":
                return EntropyHub.FuzzEn2D(X, self.m, Lock=False)
            else:
                return fuzzEntropy2D(X, self.m, self.r, self.mu)
        elif X.ndim == 1:
            if self.mode == "self":
                return fuzzEntropy(self.m, self.r, self.mu)._fuzzyEntropyCompute(X)
            elif self.mode == "hub":
                return EntropyHub.FuzzEn(X, self.m)


def phase_trials_processor(pipeline, phase):
    res = []
    for t in phase:
        if t.shape != (0, 0, 0):
            res.append(pipeline.fit_transform(t))
    return res


def all_subjects(pipeline, data):
    res = dict()
    for s in data:
        # res[s] = phase_trials_processor(pipeline,data[s]['eeg_data'])
        res[s] = pipeline.fit_transform(data[s]["eeg_data"])
        # print(s)
    print("done")
    return res
