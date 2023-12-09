import numpy as np
import mne
import copy


class filter:
    def __init__(self, lfreq, hfreq, sfreq):
        self._lfreq = lfreq
        self._hfreq = hfreq
        self._sfreq = sfreq

    def fit_transform(self, X):
        self.__method = mne.decoding.TemporalFilter(
            self._lfreq,
            self._hfreq,
            self._sfreq,
        )
        return self.__method.transform(X.T).T


class resampler:
    def __init__(self, up=1, down=1):
        self.up = down
        self.down = up

    def fit_transform(self, X, N=500):
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
        _new_data = mne.filter.resample(X, up=self.down, down=self.up)
        return _new_data


class electrode_grouper:
    def __init__(self, electrodes, target):
        self.electrodes = electrodes
        self.target = target

    def fit_transform(self, X):
        if X.ndim == 2:
            self.res = np.empty((0, X.shape[-1]))
        elif X.ndim == 3:
            self.res = np.empty((0, X.shape[-2], X.shape[-1]))
        elif X.ndim == 4:
            self.res = np.empty((0, X.shape[-3], X.shape[-2], X.shape[-1]))

        for ei, e in enumerate(self.electrodes):
            if any(i in e for i in self.target):
                if X.ndim == 2:
                    self.res = np.vstack((self.res, X[ei, :]))
                elif X.ndim == 3:
                    self.res = np.vstack(
                        (self.res, X[ei, :, :].reshape(1, X.shape[-2], X.shape[-1]))
                    )
                elif X.ndim == 4:
                    self.res = np.vstack(
                        (
                            self.res,
                            X[ei, :, :, :].reshape(
                                1, X.shape[-3], X.shape[-2], X.shape[-1]
                            ),
                        )
                    )
        return self.res


class dimension_augment_gaussian_noise:
    def __init__(self, minDim):
        self.minDim = minDim

    def fit_transform(self, X):
        if X.shape[0] < self.minDim:
            if X.ndim == 2:
                gaussian = np.random.normal(
                    0, 1, (self.minDim - X.shape[0], X.shape[1])
                )
                self.res = np.concatenate((X, gaussian), 0)
        else:
            self.res = X
        return self.res


class channel_dropper:
    def __init__(self, index, axis):
        self.axis = axis
        self.del_ = index

    def fit_transform(self, X):
        assert max(self.del_) <= X.shape[0]
        return np.delete(X, self.del_, self.axis)


class baseline_corrector:
    def __init__(self, with_std=False):
        self.with_std = False

    def fit_transform(self, X):
        mean = np.mean(X, axis=1)
        if self.with_std == True:
            std = np.std(X, axis=1)
        for c in range(X.shape[1]):
            X[:, c] = X[:, c] - mean
            if self.with_std == True:
                X[:, c] = X[:, c] / std
        return X


class time2sample:
    def __init__(self, dummy=0):
        pass

    def fit_transform(x, xm, fs):
        for i, m in enumerate(xm):
            xm[i][0] = np.floor(m[0].astype(float) * (fs[i] * 10))


class ppcn_pipeline:
    def __init__(self, processors):
        self.ppc = [0] * len(processors)
        for i in range(len(processors)):
            self.ppc[i] = processors[i][0](**processors[i][1])

    def fit_transform(self, X):
        for i in range(len(self.ppc)):
            X = self.ppc[i].fit_transform(X)
        return X


def phase_preprocessor(pipeline, phase_data):
    for i in range(len(phase_data)):
        if phase_data[i].shape != (0, 0, 0):
            phase_data[i] = pipeline.fit_transform(phase_data[i])
        else:
            phase_data[i] = np.empty((0, 0, 0))
    return phase_data


def subject_preprocessor(pipeline, subject_data):
    for p in subject_data:
        subject_data[p] = phase_preprocessor(pipeline, subject_data[p])
    return subject_data


def all_subjects_preprocessor(pipeline, subjects_data):
    data = dict()
    for s in subjects_data:
        data[s] = dict()
        data[s]["eeg_data"] = subject_preprocessor(
            pipeline, subjects_data[s]["eeg_data"]
        )
        data[s]["eeg_markers"] = subjects_data[s]["eeg_markers"]
    return data


def trials_as_subject_augmentation(data, orig_data, flag_subjects=[]):
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
