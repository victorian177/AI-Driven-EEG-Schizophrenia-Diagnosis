import copy

import mne
import numpy as np


class TemporalFilter:
    """
    Apply a temporal filter to EEG data using MNE's TemporalFilter.

    Parameters:
    - lfreq: float, the low cutoff frequency of the filter.
    - hfreq: float, the high cutoff frequency of the filter.
    - sfreq: float, the sampling frequency of the data.
    """

    def __init__(self, lfreq, hfreq, sfreq):
        self._lfreq = lfreq
        self._hfreq = hfreq
        self._sfreq = sfreq

    def fit_transform(self, X):
        """
        Fit and transform the input data using the temporal filter.

        Parameters:
        - X: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)

        Returns:
        - transformed_data: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        """
        method = mne.decoding.TemporalFilter(self._lfreq, self._hfreq, self._sfreq)
        return method.transform(X.T).T


class Resampler:
    """
    Resample EEG data using MNE's resample function.

    Parameters:
    - up: int, upsampling factor.
    - down: int, downsampling factor.
    """

    def __init__(self, up=1, down=1):
        self.up = down
        self.down = up

    def fit_transform(self, X, N=500):
        """
        Fit and transform the input data using resampling.

        Parameters:
        - X: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        - N: int, target number of samples after resampling.

        Returns:
        - resampled_data: array-like, shape (n_channels, N) or (n_epochs, n_channels, N)
        """
        n = X.shape[-1]
        if N > n:
            self.down = 1
            self.up = n / N
        elif N < n:
            self.up = 1
            self.down = N / n
        else:
            self.up = 1
            self.down = 1
        new_data = mne.filter.resample(X, up=self.down, down=self.up)
        return new_data


class ElectrodeGrouper:
    """
    Group EEG data based on specified target electrodes.

    Parameters:
    - electrodes: list, list of electrodes to include in the grouping.
    - target: list, list of target electrodes for inclusion.

    Returns:
    - grouped_data: array-like, shape (n_selected_channels, n_times) or (n_epochs, n_selected_channels, n_times)
    """

    def __init__(self, electrodes, target):
        self.electrodes = electrodes
        self.target = target

    def fit_transform(self, X):
        """
        Fit and transform the input data by selecting electrodes based on the target list.

        Parameters:
        - X: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)

        Returns:
        - grouped_data: array-like, shape (n_selected_channels, n_times) or (n_epochs, n_selected_channels, n_times)
        """
        if X.ndim == 2:
            res = np.empty((0, X.shape[-1]))
        elif X.ndim == 3:
            res = np.empty((0, X.shape[-2], X.shape[-1]))
        elif X.ndim == 4:
            res = np.empty((0, X.shape[-3], X.shape[-2], X.shape[-1]))

        for ei, e in enumerate(self.electrodes):
            if any(i in e for i in self.target):
                if X.ndim == 2:
                    res = np.vstack((res, X[ei, :]))
                elif X.ndim == 3:
                    res = np.vstack(
                        (res, X[ei, :, :].reshape(1, X.shape[-2], X.shape[-1]))
                    )
                elif X.ndim == 4:
                    res = np.vstack(
                        (
                            res,
                            X[ei, :, :, :].reshape(
                                1, X.shape[-3], X.shape[-2], X.shape[-1]
                            ),
                        )
                    )
        return res


class DimensionAugmentGaussianNoise:
    """
    Augment data dimensionality by adding Gaussian noise.

    Parameters:
    - minDim: int, minimum dimensionality required.

    Returns:
    - augmented_data: array-like, shape (minDim, n_times) or (minDim, n_channels, n_times)
    """

    def __init__(self, minDim):
        self.minDim = minDim

    def fit_transform(self, X):
        """
        Fit and transform the input data by augmenting dimensionality with Gaussian noise.

        Parameters:
        - X: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)

        Returns:
        - augmented_data: array-like, shape (minDim, n_times) or (minDim, n_channels, n_times)
        """
        if X.shape[0] < self.minDim:
            if X.ndim == 2:
                gaussian = np.random.normal(
                    0, 1, (self.minDim - X.shape[0], X.shape[1])
                )
                res = np.concatenate((X, gaussian), 0)
        else:
            res = X
        return res


class ChannelDropper:
    """
    Drop specified channels from EEG data.

    Parameters:
    - index: list, list of channel indices to drop.
    - axis: int, axis along which channels should be dropped.

    Returns:
    - modified_data: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
    """

    def __init__(self, index, axis):
        self.axis = axis
        self.del_ = index

    def fit_transform(self, X):
        """
        Fit and transform the input data by dropping specified channels.

        Parameters:
        - X: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)

        Returns:
        - modified_data: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        """
        assert max(self.del_) <= X.shape[0]
        return np.delete(X, self.del_, self.axis)


class BaselineCorrector:
    """
    Perform baseline correction on EEG data.

    Parameters:
    - with_std: bool, whether to divide by the standard deviation.

    Returns:
    - corrected_data: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
    """

    def __init__(self, with_std=False):
        self.with_std = False

    def fit_transform(self, X):
        """
        Fit and transform the input data by performing baseline correction.

        Parameters:
        - X: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)

        Returns:
        - corrected_data: array-like, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        """
        mean = np.mean(X, axis=1)
        if self.with_std:
            std = np.std(X, axis=1)
        for c in range(X.shape[1]):
            X[:, c] = X[:, c] - mean
            if self.with_std:
                X[:, c] = X[:, c] / std
        return X


class Time2Sample:
    """
    Convert time values to sample indices.

    Parameters:
    - dummy: int, a dummy parameter (not used in the current implementation).
    """

    def __init__(self, dummy=0):
        pass

    @staticmethod
    def fit_transform(x, xm, fs):
        """
        Static method to fit and transform time values to sample indices.

        Parameters:
        - x: array-like, input data.
        - xm: array-like, time values to convert.
        - fs: float, sampling frequency.

        Returns:
        - transformed_data: array-like, converted sample indices.
        """
        for i, m in enumerate(xm):
            xm[i][0] = np.floor(m[0].astype(float) * (fs[i] * 10))


class Pipeline:
    """
    Combine multiple preprocessors into a processing pipeline.

    Parameters:
    - processors: list, list of tuples containing preprocessor class and parameters.

    Returns:
    - ppc: array-like, preprocessed data.
    """

    def __init__(self, processors):
        self.ppc = [0] * len(processors)
        for i in range(len(processors)):
            self.ppc[i] = processors[i][0](**processors[i][1])

    def fit_transform(self, X):
        """
        Fit and transform the input data using the processing pipeline.

        Parameters:
        - X: array-like, input data.

        Returns:
        - processed_data: array-like, preprocessed data.
        """
        for i in range(len(self.ppc)):
            X = self.ppc[i].fit_transform(X)
        return X


def phase_preprocessor(pipeline, phase_data):
    """
    Preprocess phase data using a specified pipeline.

    Parameters:
    - pipeline: Pipeline, processing pipeline.
    - phase_data: array-like, phase data to preprocess.

    Returns:
    - preprocessed_data: array-like, preprocessed phase data.
    """
    for i in range(len(phase_data)):
        if phase_data[i].shape != (0, 0, 0):
            phase_data[i] = pipeline.fit_transform(phase_data[i])
        else:
            phase_data[i] = np.empty((0, 0, 0))
    return phase_data


def subject_preprocessor(pipeline, subject_data):
    """
    Preprocess subject data using a specified pipeline.

    Parameters:
    - pipeline: Pipeline, processing pipeline.
    - subject_data: dict, subject-specific EEG data.

    Returns:
    - preprocessed_data: dict, preprocessed subject data.
    """
    for p in subject_data:
        subject_data[p] = phase_preprocessor(pipeline, subject_data[p])
    return subject_data


def all_subjects_preprocessor(pipeline, subjects_data):
    """
    Preprocess data for all subjects using a specified pipeline.

    Parameters:
    - pipeline: Pipeline, processing pipeline.
    - subjects_data: dict, dictionary containing EEG data for multiple subjects.

    Returns:
    - preprocessed_data: dict, preprocessed data for all subjects.
    """
    data = dict()
    for s in subjects_data:
        data[s] = dict()
        data[s]["eeg_data"] = subject_preprocessor(
            pipeline, subjects_data[s]["eeg_data"]
        )
        data[s]["eeg_markers"] = subjects_data[s]["eeg_markers"]
    return data


def trials_as_subject_augmentation(data, orig_data, flag_subjects=[]):
    """
    Augment data by treating each trial as an individual subject.

    Parameters:
    - data: dict, dictionary containing EEG data.
    - orig_data: dict, original EEG data for reference.
    - flag_subjects: list, list of subjects to exclude from augmentation.

    Returns:
    - augmented_data: dict, augmented EEG data.
    """
    res = dict()
    for id in data.keys():
        if id not in flag_subjects:
            assert (
                len(data[id]["eeg_data"]["rest1"])
                == len(data[id]["eeg_data"]["rest1"])
                == len(data[id]["eeg_data"]["arith"])
                == len(data[id]["eeg_data"]["auditory"])
            )
            no_trials = len(data[id]["eeg_data"]["rest1"])
            for nt in range(no_trials):
                res[id + "_" + str(nt)] = {
                    "eeg_data": {},
                    "eeg_markers": {},
                    "category": orig_data[id]["category"],
                }
                for k in ["rest1", "rest2", "arith", "auditory"]:
                    res[id + "_" + str(nt)]["eeg_data"][k] = data[id]["eeg_data"][k][nt]
                    res[id + "_" + str(nt)]["eeg_markers"][k] = data[id]["eeg_markers"][
                        k
                    ][nt]
    return res
