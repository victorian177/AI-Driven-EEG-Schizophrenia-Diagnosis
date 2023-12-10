from math import cos, pi, radians, sin

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import griddata, interp2d

center = (0.5, 0.5)

ELECTRODES = {
    "FpZ": (0.5 + (0.33 * cos(radians(90))), 0.5 + (0.33 * sin(radians(90)))),
    "OZ": (0.5 + (0.33 * cos(radians(270))), 0.5 + (0.33 * sin(radians(270)))),
    "PZ": (0.5 + (0.16 * cos(radians(270))), 0.5 + (0.16 * sin(radians(270)))),
    "FZ": (0.5 + (0.16 * cos(radians(90))), 0.5 + (0.16 * sin(radians(90)))),
    "CZ": (0.5, 0.5),
    "Fp1": (0.5 + (0.33 * cos(radians(115))), 0.5 + (0.33 * sin(radians(115)))),
    "Fp2": (0.5 + (0.33 * cos(radians(65))), 0.5 + (0.33 * sin(radians(65)))),
    "F7": (0.5 + (0.33 * cos(radians(145))), 0.5 + (0.33 * sin(radians(145)))),
    "F8": (0.5 + (0.33 * cos(radians(35))), 0.5 + (0.33 * sin(radians(35)))),
    "T7": (0.5 + (0.33 * cos(radians(180))), 0.5 + (0.33 * sin(radians(180)))),
    "T8": (0.5 + (0.33 * cos(radians(0))), 0.5 + (0.33 * sin(radians(0)))),
    "P7": (0.5 + (0.33 * cos(radians(215))), 0.5 + (0.33 * sin(radians(215)))),
    "P8": (0.5 + (0.33 * cos(radians(-35))), 0.5 + (0.33 * sin(radians(-35)))),
    "O1": (0.5 + (0.33 * cos(radians(245))), 0.5 + (0.33 * sin(radians(245)))),
    "O2": (0.5 + (0.33 * cos(radians(-65))), 0.5 + (0.33 * sin(radians(-65)))),
    "F3": (0.5 + (0.22 * cos(radians(130))), 0.5 + (0.22 * sin(radians(130)))),
    "F4": (0.5 + (0.22 * cos(radians(50))), 0.5 + (0.22 * sin(radians(50)))),
    "P3": (0.5 + (0.22 * cos(radians(230))), 0.5 + (0.22 * sin(radians(230)))),
    "P4": (0.5 + (0.22 * cos(radians(-50))), 0.5 + (0.22 * sin(radians(-50)))),
    "C3": (0.5 + (0.16 * cos(radians(180))), 0.5 + (0.16 * sin(radians(180)))),
    "C4": (0.5 + (0.16 * cos(radians(0))), 0.5 + (0.16 * sin(radians(0)))),
}


def draw_electrode(electrodes, ax):
    for electrode in electrodes:
        # circle1 = plt.Circle(ELECTRODES[electrode],
        # radius=0.04, fill=True,
        #                         facecolor=(0, 0, 0)
        # )
        # circle2 = plt.Circle(ELECTRODES[electrode],
        #                         radius=0.03, fill=True,
        #                         facecolor=(1, 1, 1)
        # )
        # ax.add_artist(circle1)
        # ax.add_artist(circle2)

        position = ELECTRODES[electrode]
        ax.text(
            position[0],
            position[1],
            electrode,
            verticalalignment="center",
            horizontalalignment="center",
            size=6,
        )


def draw_nose(ax):
    """Draw nose."""
    nose1 = plt.Line2D([0.45, 0.5], [0.9, 0.95], linewidth=3, color=(0, 0, 0))
    nose2 = plt.Line2D([0.5, 0.55], [0.95, 0.9], linewidth=3, color=(0, 0, 0))
    ax.add_line(nose1)
    ax.add_line(nose2)


# def montage_plot(eeg_sample,electrodes,ax):
#     # assert eeg_sample.shape[0] == len(electrodes)

#     head_outer_circle = patches.Circle(center, radius=0.4, color='black')
#     head_inner_circle = patches.Circle(center, radius=0.39, color='white')

#     ax.set_aspect(1)
#     ax.add_artist(head_outer_circle)
#     ax.add_artist(head_inner_circle)

#     draw_nose(ax)
#     draw_electrode(electrodes,ax)

#     points=[]
#     X=list()
#     Y=list()
#     for electrode in electrodes:
#         X.append(ELECTRODES[electrode][0])
#         Y.append(ELECTRODES[electrode][1])
#         points.append(ELECTRODES[electrode])
#     Xi,Yi=np.meshgrid(X,Y)
#     points = np.array(points)
#     print(points.shape)

#     Z=griddata(points,eeg_sample,(Xi,Yi),'cubic')

#     ax.set_ylim(0,1)
#     ax.set_xlim(0,1)

#     ax.contour(X,Y,Z)

#     # ax.axis('equal')
#     plt.show()
