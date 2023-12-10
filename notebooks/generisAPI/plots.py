import os
from math import cos, floor, pi, radians, sin

import mne
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator, griddata, interp1d, interp2d

r1 = 0.1
r2 = 0.13
center = (r2 + 0.01, r2 + 0.01)


ELECTRODES = {
    "FpZ": (r2 + 0.01 + (r1 * cos(radians(90))), r2 + 0.01 + (r1 * sin(radians(90)))),
    "OZ": (r2 + 0.01 + (r1 * cos(radians(270))), r2 + 0.01 + (r1 * sin(radians(270)))),
    "PZ": (
        r2 + 0.01 + (0.48 * r1 * cos(radians(270))),
        r2 + 0.01 + (0.48 * r1 * sin(radians(270))),
    ),
    "FZ": (
        r2 + 0.01 + (0.48 * r1 * cos(radians(90))),
        r2 + 0.01 + (0.48 * r1 * sin(radians(90))),
    ),
    "CZ": (r2 + 0.01, r2 + 0.01),
    "Fp1": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(100))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(100))),
    ),
    "Fp2": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(80))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(80))),
    ),
    "F7": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(145))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(145))),
    ),
    "F8": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(35))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(35))),
    ),
    "T7": (r2 + 0.01 + (r1 * cos(radians(180))), r2 + 0.01 + (r1 * sin(radians(180)))),
    "T8": (r2 + 0.01 + (r1 * cos(radians(0))), r2 + 0.01 + (r1 * sin(radians(0)))),
    "P7": (r2 + 0.01 + (r1 * cos(radians(215))), r2 + 0.01 + (r1 * sin(radians(215)))),
    "P8": (r2 + 0.01 + (r1 * cos(radians(-35))), r2 + 0.01 + (r1 * sin(radians(-35)))),
    "O1": (
        r2 + 0.01 + (0.35 * r1 * cos(radians(240))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(240))),
    ),
    "O2": (
        r2 + 0.01 + (0.35 * r1 * cos(radians(-60))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(-60))),
    ),
    "F3": (
        r2 + 0.01 + (0.48 * r1 * cos(radians(90))) - (0.4 * r1),
        r2 + 0.01 + (0.48 * r1 * sin(radians(90))),
    ),
    "F4": (
        r2 + 0.01 + (0.48 * r1 * cos(radians(90))) + (0.4 * r1),
        r2 + 0.01 + (0.48 * r1 * sin(radians(90))),
    ),
    "P3": (
        r2 + 0.01 + (0.48 * r1 * cos(radians(270))) - (0.4 * r1),
        r2 + 0.01 + (0.48 * r1 * sin(radians(270))),
    ),
    "P4": (
        r2 + 0.01 + (0.48 * r1 * cos(radians(270))) + (0.4 * r1),
        r2 + 0.01 + (0.48 * r1 * sin(radians(270))),
    ),
    "C3": (
        r2 + 0.01 + (0.5 * r1 * cos(radians(180))),
        r2 + 0.01 + (0.5 * r1 * sin(radians(180))),
    ),
    "C4": (
        r2 + 0.01 + (0.5 * r1 * cos(radians(0))),
        r2 + 0.01 + (0.5 * r1 * sin(radians(0))),
    ),
    "T3": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(180))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(180))),
    ),
    "T4": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(0))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(0))),
    ),
    "T5": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(220))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(220))),
    ),
    "T6": (
        r2 + 0.01 + (0.95 * r1 * cos(radians(-40))),
        r2 + 0.01 + (0.95 * r1 * sin(radians(-40))),
    ),
}

EDGE_COORDS = [
    (r2 + 0.01 + (r2 * cos(radians(90))), r2 + 0.01 + (r2 * sin(radians(90)))),
    (r2 + 0.01 + (r2 * cos(radians(270))), r2 + 0.01 + (r2 * sin(radians(270)))),
    (r2 + 0.01 + (r2 * cos(radians(115))), r2 + 0.01 + (r2 * sin(radians(115)))),
    (r2 + 0.01 + (r2 * cos(radians(65))), r2 + 0.01 + (r2 * sin(radians(65)))),
    (r2 + 0.01 + (r2 * cos(radians(145))), r2 + 0.01 + (r2 * sin(radians(145)))),
    (r2 + 0.01 + (r2 * cos(radians(35))), r2 + 0.01 + (r2 * sin(radians(35)))),
    (r2 + 0.01 + (r2 * cos(radians(180))), r2 + 0.01 + (r2 * sin(radians(180)))),
    (r2 + 0.01 + (r2 * cos(radians(0))), r2 + 0.01 + (r2 * sin(radians(0)))),
    (r2 + 0.01 + (r2 * cos(radians(215))), r2 + 0.01 + (r2 * sin(radians(215)))),
    (r2 + 0.01 + (r2 * cos(radians(-35))), r2 + 0.01 + (r2 * sin(radians(-35)))),
    (r2 + 0.01 + (r2 * cos(radians(245))), r2 + 0.01 + (r2 * sin(radians(245)))),
    (r2 + 0.01 + (r2 * cos(radians(-65))), r2 + 0.01 + (r2 * sin(radians(-65)))),
]


def draw_electrode(electrodes, ax):
    for electrode in electrodes:
        position = ELECTRODES[electrode]
        ax.text(
            position[0],
            position[1],
            electrode,
            verticalalignment="center",
            horizontalalignment="center",
            size=10,
        )


def draw_nose(ax):
    """Draw nose."""
    nose1 = plt.Line2D([0.13, 0.14], [0.27, 0.29], linewidth=1, color=(0, 0, 0))
    nose2 = plt.Line2D([0.14, 0.15], [0.29, 0.27], linewidth=1, color=(0, 0, 0))
    ax.add_line(nose1)
    ax.add_line(nose2)


def montage_plot(eeg_sample, electrodes, ax):
    head_outer_circle = patches.Circle(center, radius=r2 + 0.001, color="black")
    head_inner_circle = patches.Circle(center, radius=r2, color="white")

    # ax.set_aspect(1)
    ax.add_artist(head_outer_circle)
    ax.add_artist(head_inner_circle)

    draw_nose(ax)
    draw_electrode(electrodes, ax)

    points = []
    for electrode in electrodes:
        points.append(ELECTRODES[electrode])
    points = points + EDGE_COORDS
    X, Y = np.mgrid[0:1:50j, 0:1:50j]
    Xi, Yi = np.meshgrid(X, Y)
    points = np.array(points)

    mu = np.mean(eeg_sample, 0)
    mu_add = np.ones((len(EDGE_COORDS))) * mu
    data_sample = np.hstack((eeg_sample, mu_add))
    # data_sample = eeg_sample

    ax.set_ylim(0, 2 * (r2 + 0.02))
    ax.set_xlim(0, 2 * (r2 + 0.02))

    Z = griddata(points, data_sample, (Xi, Yi), "cubic")
    CS = ax.contour(
        Xi, Yi, Z, 30, cmap="RdBu", extend="both", linewidths=20, extent=(0, 1, 0, 1)
    )

    # ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.patch.set_facecolor((1, 1, 1, 0.01))
    ax.patch.set_alpha(0.0)
    # ax.set_aspect('equal')
    # plt.show()


def stft_plot(ax, data, title, fs=9, sfreq=200):
    f = mne.time_frequency.stftfreq((fs - 1) * 2, sfreq)
    ax.pcolormesh(list(range(data.shape[-1])), f, np.abs(data), shading="gouraud")
    ax.set_xlabel("Time(ms)")
    ax.set_ylabel("Frequency(Hz)")
    ax.set_title(title)


def time_series_plot(ax, data, color, stim, xi, x, c):
    ax.plot(
        data,
        color,
        label=stim,
        linestyle="-",
    )
    ax.set_xticks(xi, x)
    # ax.set_xlabel('Time(s)')
    ax.set_title(c)


def mmn_plot(ax, data, ch, points=15):
    xi = list(range(0, data["1000Hz"].shape[1], points))
    x = (np.array(xi) / 200).tolist()
    for r in range(data["1000Hz"].shape[0]):
        for st, stim in enumerate(data):
            if stim == "1000Hz":
                color = "#d7d0cf"
                time_series_plot(ax[r], data[stim][r, :], color, stim, xi, x, ch[r])
            elif stim == " ":
                color = "#868483"
            else:
                color = "#3b3a3a"
                time_series_plot(ax[r], data[stim][r, :], color, stim, xi, x, ch[r])
        if r == 0:
            # ax[r].legend(loc='upper left',fontsize=6)
            ax[r].set_ylabel("Amplitude(uV)")
        if r != 0:
            ax[r].set_ylabel(" ")


def electrodes_mmn_plot(ax, data, c0, c1, electrodes, points):
    mmn_data = {
        " ": data[" "][c0:c1, :],
        "1000Hz": data["1000Hz"][c0:c1, :],
        "3000Hz": data["3000Hz"][c0:c1, :],
    }
    mmn_plot(ax, mmn_data, electrodes[c0:c1], points)


def mmn_feature_plot(
    mmn_value1_array, mmn_value2_array, patients, controls, electrodes, div, title
):
    fig, ax = plt.subplots(3, 7, figsize=(16, 6))
    for r in range(3):
        for c in range(7):
            ax[r, c].scatter(
                mmn_value1_array[patients, div, (7 * r) + c],
                mmn_value2_array[patients, div, (7 * r) + c],
            )
            ax[r, c].scatter(
                mmn_value1_array[controls, div, (7 * r) + c],
                mmn_value2_array[controls, div, (7 * r) + c],
            )
            ax[r, c].set_title(electrodes[(7 * r) + c])
            if r == 2:
                ax[r, c].set_xlabel("1k_duration_deviant")
            if c == 0:
                ax[r, c].set_ylabel("3k_duration_deviant")
            if (7 * r) + c == 18:
                break
    fig.suptitle(title)
    fig.tight_layout(pad=0.95)
    fig.legend(["Patients", "Controls"], loc="lower right")


def entropy_plot(ax, data, pos):
    if len(data) > 2:
        data = np.array(data).mean()
    ax.bar(pos, data, 1)
    # ax.set_xticks([1,2,3,4],['rest1','arith','rest2','auditory'])
    # ax.set_ylabel('FuzzEnt')


def get_assr_data_4_plot(assr):
    assrs = np.empty((len(assr), 2, 19))
    for si, subject in enumerate(assr):
        assrs[si, :, :] = assr[subject]
    return assrs


def assr_plot(patients, controls, electrodes, title="ASSR plot"):
    fig, ax = plt.subplots(3, 7, figsize=(14, 6))
    for ei, electrode in enumerate(electrodes):
        r = floor(ei / 7)
        c = ei % 7
        ax[r, c].scatter(patients[:, 1, ei], patients[:, 0, ei])
        ax[r, c].scatter(controls[:, 1, ei], controls[:, 0, ei])
        ax[r, c].set_title(electrode)
        if c == 0:
            ax[r, c].set_ylabel("Amplitude")
        if r == 2:
            ax[r, c].set_xlabel("Phase")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.legend(["patients", "controls"], loc="lower right")
    fig.suptitle(title)
