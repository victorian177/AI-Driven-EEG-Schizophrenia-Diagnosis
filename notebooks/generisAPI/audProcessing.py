import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale


# def convertMarkersTime2Sample(data,subject,phase):
#     x=data[subject]['eeg_data'][phase]
#     xm=data[subject]['eeg_markers'][phase]
#     fs= data[subject]['Fs'][phase]
#     for i,m in enumerate(xm):
#         xm[i][0] = np.floor(m[0].astype(float)*(fs[i]*10))s


def auditory_stimuli_epoching(data, subject, dt):
    dt = int(dt * 200)
    x = data[subject]["eeg_data"]
    xm = data[subject]["eeg_markers"]
    res = dict()
    if isinstance(x, list):
        for p, pdata in enumerate(xm):
            if x[p].shape != (0, 0, 0):
                unique_markers = np.unique(pdata[1])
                unique_index = [np.where(pdata[1] == m) for m in unique_markers]
                for m in unique_markers:
                    res[m] = np.empty((0, x[p].shape[0], dt))
                for m in range(len(unique_markers)):
                    for i, ind in enumerate(unique_index[m][0]):
                        res[unique_markers[m]] = np.vstack(
                            (
                                res[unique_markers[m]],
                                x[p][:, ind : ind + dt].reshape(1, x[p].shape[0], dt),
                            )
                        )
    elif isinstance(x, np.ndarray):
        if x.shape != (0, 0, 0):
            unique_markers = np.unique(xm[1])
            unique_index = [np.where(xm[1] == m) for m in unique_markers]
            for m in unique_markers:
                res[m] = np.empty((0, x.shape[0], dt))
            for m in range(len(unique_markers)):
                for i, ind in enumerate(unique_index[m]):
                    for ts in ind:
                        res[unique_markers[m]] = np.vstack(
                            (
                                res[unique_markers[m]],
                                x[:, ts : ts + dt].reshape(1, x.shape[0], dt),
                            )
                        )
    return res


def subject_phase_auditory_epochs(data, subject, dt):
    res = auditory_stimuli_epoching(data, subject, dt)
    return res


def all_subjects_phase_auditory_epochs(data, dt):
    res = dict()
    for subject in data:
        res[subject] = subject_phase_auditory_epochs(data, subject, dt)
    return res


class pipeline:
    def __init__(self, processors):
        self.ppc = [0] * len(processors)
        for i in range(len(processors)):
            self.ppc[i] = processors[i][0](**processors[i][1])

    def fit_transform(self, X):
        for i in range(len(self.ppc)):
            X = self.ppc[i].fit_transform(X)
        return X


class epoch_std:
    def __init__(self):
        pass

    def fit_transform(self, X):
        scl = StandardScaler()
        res = np.empty(X.shape)
        if X.ndim == 3:
            for i in range(X.shape[0]):
                res[i, :, :] = scl.fit_transform(X[i, :, :])
        elif X.ndim == 2:
            res = scl.fit_transform(X)
        return res


class epoch_MinMax:
    def __init__(self, range, axis):
        self.range = range
        self.axis = axis

    def fit_transform(self, X):
        res = np.empty(X.shape)
        if X.ndim == 3:
            for i in range(X.shape[0]):
                res[i, :, :] = minmax_scale(
                    X[i, :, :], self.range, feature_range=self.range, axis=self.axis
                )
        elif X.ndim == 2:
            res = minmax_scale(X, feature_range=self.range, axis=self.axis)
        return res


class aud_stimuli_trial_average:
    def __init_(self):
        pass

    def fit_transform(self, X):
        return np.average(X, axis=0)


class stimuli_mmn:
    def __init__(self, std_tone, N):
        self.std_tone = std_tone
        self.n = N

    def fit_transform(self, X):
        res = dict()
        for s in X:
            temp_res = dict.fromkeys(X[s].keys())
            for stim in X[s]:
                x = X[s][stim] - X[s][self.std_tone]
                n = self.n
                ret = np.cumsum(x, 1)
                ret[:, n:] = ret[:, n:] - ret[:, :-n]
                temp_res[stim] = ret[:, n - 1 :] / n

            res[s] = temp_res
        return res


def phase_processor(pipeline, phase_data):
    res = dict()
    for stim in phase_data:
        res[stim] = pipeline.fit_transform(phase_data[stim])
    return res


def all_subjects_processor(pipeline, data):
    res = dict()
    for subject in data:
        res[subject] = phase_processor(pipeline, data[subject])
    return res
