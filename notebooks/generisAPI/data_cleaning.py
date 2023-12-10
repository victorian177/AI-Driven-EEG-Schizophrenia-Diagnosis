import copy

import numpy as np


class assertTrialsMarkersLength:
    def __init__(self):
        self.indices = list()

    def fit(self, X):
        data = X
        self.indices = list()
        for subject in data:
            for mode in data[subject]["eeg_data"]:
                if len(data[subject]["eeg_data"][mode]) != len(
                    data[subject]["eeg_markers"][mode]
                ):
                    self.indices.append([subject, mode])
        return self

    def transform(self, X, inplace=False):
        if inplace == True:
            data = X
        else:
            data = copy.deepcopy(X)
        for index in self.indices:
            data[index[0]]["eeg_data"][index[1]] = [np.empty((0, 0, 0))]
            data[index[0]]["eeg_markers"][index[1]] = [np.empty((0, 0, 0))]
        return data

    def fit_transform(self, X, inplace=False):
        self.fit(X)
        return self.transform(X, inplace=inplace)


class nullIndicesTracer:
    def __init__(self):
        self.indices = dict()

    def fit(self, X):
        data = X
        self.indices = dict()
        for subject in data:
            s = dict()
            for mode in data[subject]["eeg_data"]:
                null = False
                l = list()
                for i, arr in enumerate(data[subject]["eeg_data"][mode]):
                    if arr.shape == (0, 0, 0):
                        l.append(i)
                        null = True
                s[mode] = l
            if null == True:
                self.indices[subject] = s
        return self

    def transform(self, X, inplace=False):
        if inplace == True:
            data = X
        else:
            data = copy.deepcopy(X)
        for subject in self.indices:
            for mode in self.indices[subject]:
                for i in self.indices[subject][mode]:
                    data[subject]["eeg_markers"][mode][i] = np.empty((0, 0))
        return data

    def fit_transform(self, X, inplace=False):
        self.fit(X)
        return self.transform(X, inplace=inplace)
