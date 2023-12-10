import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale


def auditory_stimuli_epoching(data, subject, dt):
    """
    Perform epoching on auditory stimuli EEG data.

    Args:
        data (dict): Dictionary containing EEG data.
        subject (str): Key representing the subject in the data dictionary.
        dt (float): Time duration of each epoch in seconds.

    Returns:
        dict: Dictionary containing epoch data for each stimulus marker.
    """
    dt = int(dt * 200)  # Convert time duration to samples
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
    Process auditory stimuli EEG data for a single subject.

    Args:
        data (dict): Dictionary containing EEG data.
        subject (str): Key representing the subject in the data dictionary.
        dt (float): Time duration of each epoch in seconds.

    Returns:
        dict: Dictionary containing epoch data for each stimulus marker.
    """
    res = auditory_stimuli_epoching(data, subject, dt)
    return res


def all_subjects_phase_auditory_epochs(data, dt):
    """
    Process auditory stimuli EEG data for all subjects.

    Args:
        data (dict): Dictionary containing EEG data for multiple subjects.
        dt (float): Time duration of each epoch in seconds.

    Returns:
        dict: Dictionary containing epoch data for each subject and stimulus marker.
    """
    res = dict()
    for subject in data:
        res[subject] = subject_phase_auditory_epochs(data, subject, dt)
    return res


class Pipeline:
    """
    A class representing a data processing pipeline.

    Attributes:
        ppc (list): List of processor instances in the pipeline.
    """

    def __init__(self, processors):
        """
        Initialize the pipeline with a list of processors.

        Args:
            processors (list): List of tuples, each containing a processor class
                               and its initialization parameters.
        """
        self.ppc = [0] * len(processors)
        for i in range(len(processors)):
            self.ppc[i] = processors[i][0](**processors[i][1])

    def fit_transform(self, X):
        """
        Fit and transform the input data using the pipeline.

        Args:
            X: Input data to be processed.

        Returns:
            Processed data.
        """
        for i in range(len(self.ppc)):
            X = self.ppc[i].fit_transform(X)
        return X


class EpochStd:
    """
    Standardize EEG epoch data using scikit-learn's StandardScaler.
    """

    def __init__(self):
        pass

    def fit_transform(self, X):
        """
        Fit and transform the input data using standardization.

        Args:
            X: Input data to be standardized.

        Returns:
            Standardized data.
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
    Scale EEG epoch data using scikit-learn's MinMaxScaler.
    """

    def __init__(self, range, axis):
        self.range = range
        self.axis = axis

    def fit_transform(self, X):
        """
        Fit and transform the input data using Min-Max scaling.

        Args:
            X: Input data to be scaled.

        Returns:
            Scaled data.
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
    Compute the trial average of auditory stimuli EEG data.
    """

    def __init__(self):
        pass

    def fit_transform(self, X):
        """
        Compute the trial average of the input data.

        Args:
            X: Input data.

        Returns:
            Trial-averaged data.
        """
        return np.average(X, axis=0)


class StimuliMMN:
    """
    Compute the Mismatch Negativity (MMN) of EEG data.

    Attributes:
        std_tone (str): The standard tone used as a reference.
        n (int): The window size for computing MMN.
    """

    def __init__(self, std_tone, N):
        self.std_tone = std_tone
        self.n = N

    def fit_transform(self, X):
        """
        Compute MMN for EEG data.

        Args:
            X: Input data.

        Returns:
            MMN-computed data.
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
    Process phase data using a given pipeline.

    Args:
        pipeline (Pipeline): An instance of the data processing pipeline.
        phase_data (pd.DataFrame): Phase data to be processed.

    Returns:
        pd.DataFrame: Processed phase data.
    """
    X = pipeline.fit_transform(phase_data.values)
    return pd.DataFrame(data=X, columns=phase_data.columns)


def montage_plot(eeg_sample, electrodes, ax):
    # assert eeg_sample.shape[0] == len(electrodes)

    head_outer_circle = patches.Circle(center, radius=0.4, color="black")
    head_inner_circle = patches.Circle(center, radius=0.39, color="white")

    ax.set_aspect(1)
    ax.add_artist(head_outer_circle)
    ax.add_artist(head_inner_circle)

    draw_nose(ax)
    draw_electrode(electrodes, ax)

    points = []
    X = list()
    Y = list()
    for electrode in electrodes:
        X.append(ELECTRODES[electrode][0])
        Y.append(ELECTRODES[electrode][1])
        points.append(ELECTRODES[electrode])
    Xi, Yi = np.meshgrid(X, Y)
    points = np.array(points)
    print(points.shape)

    Z = griddata(points, eeg_sample, (Xi, Yi), "cubic")

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    ax.contour(X, Y, Z)

    # ax.axis('equal')
    plt.show()
