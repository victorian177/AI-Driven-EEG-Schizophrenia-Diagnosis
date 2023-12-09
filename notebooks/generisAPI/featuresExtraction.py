import numpy as np
import scipy.signal as sig


class MMN:
    def __init__(self, delay=10, position=40, window_size=20):
        self.d = delay
        self.p = position
        self.w = window_size

    def mmn_value(self, x):
        if x.ndim == 1:
            return np.mean(x[self.p - self.d : self.p - self.d + self.w])
        elif x.ndim == 2:
            return np.mean(x[:, self.p - self.d : self.p - self.d + self.w], axis=1)


class ASSR:
    def __init__(self, freq, dfreq, fs=200):
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
    new_assr_data = dict()
    for s in assr_data:
        if s not in null_data_checker.indices:
            new_assr_data[s] = pcn.trials_averaging().fit_transform(assr_data[s])
    return new_assr_data


# def mmn_value(time_series, deviant_position, standard_position, window_size, delay):
#     """
#     Computes the Mismatch Negativity (MMN) value from a time series
#     :param time_series: 1D numpy array containing the time series data
#     :param deviant_position: the position of the deviant stimulus in the time series
#     :param standard_position: the position of the standard stimulus in the time series
#     :param window_size: the size of the temporal window used for analysis in milliseconds
#     :param delay: the delay in milliseconds between the deviant and standard stimuli
#     :return: the MMN value
#     """
#     deviant_window = time_series[deviant_position-delay:deviant_position-delay+window_size]
#     standard_window = time_series[standard_position-delay:standard_position-delay+window_size]
#     mmn = np.mean(deviant_window) - np.mean(standard_window)
#     return mmn
