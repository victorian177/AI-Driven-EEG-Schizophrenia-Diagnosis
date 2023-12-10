import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots


class MMNFeaturesAnalysis:
    """
    Class for analyzing Mismatch Negativity (MMN) features.
    """

    def __init__(self, electrodes):
        """
        Constructor for MMNFeaturesAnalysis.

        Args:
            electrodes (list): List of electrode names.
        """
        self.electrodes = electrodes

    def compare_mmn_deviants_frame(
        self,
        deviant1,
        deviant2,
        classes,
        frame,
        title,
        name,
    ):
        """
        Compare MMN deviants between classes and create a scatter plot.

        Args:
            deviant1 (numpy.ndarray): Data for the first deviant.
            deviant2 (numpy.ndarray): Data for the second deviant.
            classes (pandas.Series): Classes for each data point.
            frame (int): Frame index.
            title (str): Plot title.
            name (str): Filename to save the plot.
        """
        plt.ioff()
        fig, ax = plt.subplots(3, 7, figsize=(18, 8))
        fig.suptitle(title)
        f = frame
        for ch in range(19):
            r = math.floor(ch / 7)
            c = ch % 7
            if c == 0:
                ax[r, c].set_ylabel("1k_duration_deviant")
            ax[r, c].set_xlabel("3k_freq_deviant")
            ax[r, c].set_title(self.electrodes[ch])
            mmn_df = pd.DataFrame(
                np.vstack([deviant1[:, f, ch], deviant2[:, f, ch]]).T,
                columns=["1k", "3k"],
            )
            mmn_df["category"] = classes.to_list()
            patients = mmn_df.loc[mmn_df["category"] == "Patient"]
            controls = mmn_df.loc[mmn_df["category"] == "Control"]
            ax[r, c].scatter(patients["1k"], patients["3k"], label="Patients")
            ax[r, c].scatter(controls["1k"], controls["3k"], label="Controls")
            ax[r, c].legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.91)
        plt.savefig(name, format="png")
        plt.close(fig)


class EntropyFeaturesAnalysis:
    """
    Class for analyzing entropy features.
    """

    def __init__(self):
        pass

    def compare_entropies_between_classes(
        self,
        df,
        patients_index,
        controls_index,
        title="title",
        name="name",
        size=(20, 20),
    ):
        """
        Compare entropies between classes and create scatter plots.

        Args:
            df (pandas.DataFrame): DataFrame containing entropy values.
            patients_index (list): Indices of patients in the DataFrame.
            controls_index (list): Indices of controls in the DataFrame.
            title (str): Plot title.
            name (str): Filename to save the plot.
            size (tuple): Figure size.
        """
        ch = df.shape[1]
        plt.ion()
        fig, ax = plt.subplots(ch, ch, figsize=size)
        fig.suptitle(title)
        for r in range(ch):
            for c in range(ch):
                y_vals = df.iloc[patients_index, c]
                x_vals = df.iloc[patients_index, r]
                ax[r, c].scatter(x_vals, y_vals, label="Patients")
                y_vals = df.iloc[controls_index, c]
                x_vals = df.iloc[controls_index, r]
                ax[r, c].scatter(x_vals, y_vals, label="Controls")
                ax[r, c].legend()
                if c == 0:
                    ax[r, c].set_ylabel(df.columns[r])
                if r == ch - 1:
                    ax[r, c].set_xlabel(df.columns[c])
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.savefig(name, format="png")
        plt.close(fig)

    def correlation_matrix(self, df):
        """
        Compute and plot the correlation matrix of a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing entropy values.
        """
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        plt.imshow(correlation_matrix, cmap="viridis", interpolation="none")
        plt.colorbar()
        plt.title("Correlation Matrix")
        plt.xticks(
            range(len(correlation_matrix.columns)),
            correlation_matrix.columns,
            rotation=90,
        )
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.show()
