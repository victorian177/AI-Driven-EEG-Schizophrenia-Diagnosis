import numpy as np
import scipy.signal as sig


class MMN:
    """
    Mismatch Negativity (MMN) class for computing MMN values from time series data.
    """

    def __init__(self, delay=10, position=40, window_size=20):
        """
        Constructor for the MMN class.

        Args:
            delay (int): The delay in samples between the deviant and standard stimuli.
            position (int): The position of the deviant stimulus in the time series.
            window_size (int): The size of the temporal window used for analysis in samples.
        """
        self.d = delay
        self.p = position
        self.w = window_size

    def mmn_value(self, x):
        """
        Computes the MMN value from a time series.

        Args:
            x (numpy.ndarray): 1D or 2D numpy array containing the time series data.

        Returns:
            numpy.ndarray: The MMN values.
        """
        if x.ndim == 1:
            return np.mean(x[self.p - self.d : self.p - self.d + self.w])
        elif x.ndim == 2:
            return np.mean(x[:, self.p - self.d : self.p - self.d + self.w], axis=1)


class ASSR:
    """
    Auditory Steady State Response (ASSR) class for computing ASSR values from time series data.
    """

    def __init__(self, freq, dfreq, fs=200):
        """
        Constructor for the ASSR class.

        Args:
            freq (float): The stimulation frequency.
            dfreq (float): The frequency deviation.
            fs (int): The sampling frequency.
        """
        self.freq = freq
        self.lfreq = freq - dfreq
        self.hfreq = freq + dfreq
        self.fs = fs
        self.nyquist = fs / 2
        self.b, self.a = sig.butter(
            4, [self.lfreq / self.nyquist, self.hfreq / self.nyquist], btype="band"
        )
        self.fit_transform = self.assr

    def assr(self, X):
        """
        Computes the ASSR values from time series data.

        Args:
            X (numpy.ndarray): 1D or 2D numpy array containing the time series data.

        Returns:
            tuple: A tuple containing ASSR amplitude and phase values.
        """
        # reshape data if need be
        if X.ndim == 1:
            X = X.reshape(1, X.shape[0])

        # filter data in frequency range of interest
        filtered_data = sig.filtfilt(self.b, self.a, X, axis=1)

        # generate complex sinusoids at stimulation frequency for each channel
        n_samples = filtered_data.shape[1]
        t = np.arange(n_samples) / self.fs
        stim_complex = np.exp(2j * np.pi * self.freq * t)
        stim_complex = np.tile(stim_complex, (filtered_data.shape[0], 1))

        # perform frequency domain averaging to generate ASSR
        hilbert_data = sig.hilbert(filtered_data, axis=1)
        assr_complex = np.mean(stim_complex * hilbert_data, axis=1)
        assr_amplitude = np.abs(assr_complex)
        assr_phase = np.angle(assr_complex)

        return assr_amplitude, assr_phase


def average_trials_assr(assr_data, null_data_checker):
    """
    Averages ASSR values across trials.

    Args:
        assr_data (dict): Dictionary containing ASSR data for each subject.
        null_data_checker: An object providing information about null data.

    Returns:
        dict: Dictionary containing averaged ASSR data for each subject.
    """
    new_assr_data = dict()
    for s in assr_data:
        if s not in null_data_checker.indices:
            new_assr_data[s] = pcn.trials_averaging().fit_transform(assr_data[s])
    return new_assr_data
