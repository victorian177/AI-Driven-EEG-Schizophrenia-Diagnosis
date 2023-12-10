import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale


def auditory_stimuli_epoching(data, subject, dt):
    """
    Epoch auditory stimuli data based on unique markers.

    Args:
        data (dict): Dictionary containing EEG data and markers.
        subject (str): Subject identifier.
        dt (float): Time duration in seconds for each epoch.

    Returns:
        dict: Dictionary containing auditory stimuli epochs for each unique marker.
    """
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
    """
    Generate auditory epochs for a specific subject.

    Args:
        data (dict): Dictionary containing EEG data and markers for all subjects.
        subject (str): Subject identifier.
        dt (float): Time duration in seconds for each epoch.

    Returns:
        dict: Dictionary containing auditory stimuli epochs for each unique marker.
    """
    res = auditory_stimuli_epoching(data, subject, dt)
    return res


def all_subjects_phase_auditory_epochs(data, dt):
    """
    Generate auditory epochs for all subjects.

    Args:
        data (dict): Dictionary containing EEG data and markers for all subjects.
        dt (float): Time duration in seconds for each epoch.

    Returns:
        dict: Dictionary of dictionaries containing auditory stimuli epochs for each unique marker for all subjects.
    """
    res = dict()
    for subject in data:
        res[subject] = subject_phase_auditory_epochs(data, subject, dt)
    return res


class Pipeline:
    """
    Class for creating a processing pipeline for EEG data.
    """

    def __init__(self, processors):
        """
        Constructor for Pipeline.

        Args:
            processors (list): List of tuples containing processing class and its parameters.
        """
        self.ppc = [0] * len(processors)
        for i in range(len(processors)):
            self.ppc[i] = processors[i][0](**processors[i][1])

    def fit_transform(self, X):
        """
        Fit and transform the input data through the processing pipeline.

        Args:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        for i in range(len(self.ppc)):
            X = self.ppc[i].fit_transform(X)
        return X


class EpochStd:
    """
    Class for standardizing EEG epochs.
    """

    def __init__(self):
        pass

    def fit_transform(self, X):
        """
        Fit and transform EEG epochs using standard scaling.

        Args:
            X (numpy.ndarray): EEG epochs.

        Returns:
            numpy.ndarray: Standardized EEG epochs.
        """
        scl = StandardScaler()
        res = np.empty(X.shape)
        if X.ndim == 3:
            for i in range(X.shape[0]):
                res[i, :, :] = scl.fit_transform(X[i, :, :])
        elif X.ndim == 2:
            res = scl.fit_transform(X)
        return res


class EpochMinMax:
    """
    Class for scaling EEG epochs using Min-Max scaling.
    """

    def __init__(self, range, axis):
        self.range = range
        self.axis = axis

    def fit_transform(self, X):
        """
        Fit and transform EEG epochs using Min-Max scaling.

        Args:
            X (numpy.ndarray): EEG epochs.

        Returns:
            numpy.ndarray: Scaled EEG epochs.
        """
        res = np.empty(X.shape)
        if X.ndim == 3:
            for i in range(X.shape[0]):
                res[i, :, :] = minmax_scale(
                    X[i, :, :], self.range, feature_range=self.range, axis=self.axis
                )
        elif X.ndim == 2:
            res = minmax_scale(X, feature_range=self.range, axis=self.axis)
        return res


class AudStimuliTrialAverage:
    """
    Class for averaging auditory stimuli trials.
    """

    def __init__(self):
        pass

    def fit_transform(self, X):
        """
        Fit and transform averaged auditory stimuli trials.

        Args:
            X (numpy.ndarray): Auditory stimuli trials.

        Returns:
            numpy.ndarray: Averaged auditory stimuli trials.
        """
        return np.average(X, axis=0)


class StimuliMMN:
    """
    Class for computing stimuli Mismatch Negativity (MMN).
    """

    def __init__(self, std_tone, N):
        self.std_tone = std_tone
        self.n = N

    def fit_transform(self, X):
        """
        Fit and transform stimuli MMN.

        Args:
            X (dict): Dictionary containing auditory stimuli epochs.

        Returns:
            dict: Dictionary containing computed MMN for each subject.
        """
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
    """
    Process EEG phase data using a pipeline.

    Args:
        pipeline (Pipeline): Processing pipeline.
        phase_data (dict): Dictionary containing EEG phase data for different stimuli.

    Returns:
        dict: Dictionary containing processed EEG phase data for different stimuli.
    """
    res = dict()
    for stim in phase_data:
        res[stim] = pipeline.fit_transform(phase_data[stim])
    return res


def all_subjects_processor(pipeline, data):
    """
    Process EEG data for all subjects using a pipeline.

    Args:
        pipeline (Pipeline): Processing pipeline.
        data (dict): Dictionary containing EEG data for all subjects.

    Returns:
        dict: Dictionary containing processed EEG data for all subjects.
    """
    res = dict()
    for subject in data:
        res[subject] = phase_processor(pipeline, data[subject])
    return res
