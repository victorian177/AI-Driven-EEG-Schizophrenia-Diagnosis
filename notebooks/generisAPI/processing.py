import math

import EntropyHub
import mne
import numpy as np
from generisAPI.fuzzy_entropy import *


def xtract_phase_data(phase, data):
    """
    Extract data for a specific phase.

    Parameters:
    - phase: str, phase name
    - data: dict, EEG data

    Returns:
    - dict, EEG data for the specified phase
    """
    res = dict()
    for s in data:
        res[s] = dict()
        res[s]["eeg_data"] = data[s]["eeg_data"][phase]
        res[s]["eeg_markers"] = data[s]["eeg_markers"][phase]
    return res


class TrialEpoching:
    """
    Epoch EEG data into trials.

    Parameters:
    - mode: str, epoching mode ("min" or "max")
    """

    def __init__(self, mode="min"):
        self.mode = mode

    def fit_transform(self, X):
        """
        Fit and transform the data by epoching.

        Parameters:
        - X: ndarray, EEG data

        Returns:
        - ndarray, epoched EEG data
        """
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


class TrialAveraging:
    """
    Average EEG trials.

    Parameters:
    - axis: int, axis for averaging
    - dummy: int, dummy parameter
    """

    def __init__(self, axis=0, dummy=0):
        self.axis = axis

    def fit_transform(self, X):
        """
        Fit and transform the data by averaging.

        Parameters:
        - X: ndarray, EEG data

        Returns:
        - ndarray, averaged EEG data
        """
        return np.average(X, self.axis)


class TrialsAveraging:
    """
    Average EEG trials across subjects.

    Parameters:
    - None
    """

    def __init__(self):
        pass

    def fit_transform(self, X):
        """
        Fit and transform the data by averaging across trials.

        Parameters:
        - X: list, list of EEG data arrays

        Returns:
        - ndarray, averaged EEG data
        """
        sum_data = 0
        for x in X:
            sum_data += np.array(x)
        avg_data = sum_data / len(X)
        return avg_data


class STFT:
    """
    Apply Short-Time Fourier Transform (STFT) to EEG data.

    Parameters:
    - fs: int, size of FFT window
    - sfreq: float, sampling frequency
    """

    def __init__(self, fs, sfreq):
        self.fs = fs
        self.w = (fs - 1) * 2
        self.f = mne.time_frequency.stftfreq(self.w, sfreq)

    def __stft(self, x):
        return mne.time_frequency.stft(x, self.w)

    def fit_transform(self, X):
        """
        Fit and transform the data using STFT.

        Parameters:
        - X: ndarray, EEG data

        Returns:
        - ndarray, STFT-transformed EEG data
        """
        if X.ndim == 2:
            return self.__stft(X)
        elif X.ndim == 3:
            res = np.empty(
                (X.shape[0], X.shape[1], self.fs, math.ceil(X.shape[2] * 2 / self.w))
            )
            for i in range(X.shape[0]):
                res[i, :, :, :] = self.__stft(X[i, :, :])
        return res


class FuzzEnt:
    """
    Compute Fuzzy Entropy for EEG data.

    Parameters:
    - m: int, embedding dimension
    - r: float, tolerance parameter
    - mode: str, entropy calculation mode ("self" or "hub")
    - muFunction: str, membership function ("linear" or other)

    Note: The 'mode' parameter defines whether to use 'self' or 'hub' for entropy calculation.
    """

    def __init__(self, m, r=1, mode="self", muFunction="linear"):
        self.m = m
        self.r = r
        self.mu = muFunction
        self.mode = mode

    def fit_transform(self, X):
        """
        Fit and transform the data using Fuzzy Entropy.

        Parameters:
        - X: ndarray, EEG data

        Returns:
        - ndarray, Fuzzy Entropy-transformed data
        """
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
    """
    Process EEG trials for a specific phase.

    Parameters:
    - pipeline: object, EEG preprocessing pipeline
    - phase: list of ndarray, EEG data for each trial

    Returns:
    - list of ndarray, preprocessed EEG data for each trial
    """
    res = []
    for t in phase:
        if t.shape != (0, 0, 0):
            res.append(pipeline.fit_transform(t))
    return res


def all_subjects(pipeline, data):
    """
    Preprocess EEG data for all subjects.

    Parameters:
    - pipeline: object, EEG preprocessing pipeline
    - data: dict, EEG data for all subjects

    Returns:
    - dict, preprocessed EEG data for all subjects
    """
    res = dict()
    for s in data:
        res[s] = pipeline.fit_transform(data[s]["eeg_data"])
    print("done")
    return res
