import numpy as np
import matplotlib.pyplot as plt
import plots
import math
import pandas as pd


class mmnFeaturesAnalysis:
    def __init__(self, electrodes):
        self.electrodes = electrodes
        pass

    def compareMMNdevaintsFrame(self, deviant1, deviant2, classes, frame, title, name):
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


class entropyFeaturesAnalysis:
    def __init__(self):
        pass

    def compareEntropiesBtwClasses(
        self,
        df,
        patientsIndex,
        controlsIndex,
        title="title",
        name="name",
        size=(20, 20),
    ):
        ch = df.shape[1]
        plt.ion()
        fig, ax = plt.subplots(ch, ch, figsize=size)
        fig.suptitle(title)
        for r in range(ch):
            for c in range(ch):
                y_vals = df.iloc[patientsIndex, c]
                x_vals = df.iloc[patientsIndex, r]
                ax[r, c].scatter(x_vals, y_vals, label="Patients")
                y_vals = df.iloc[controlsIndex, c]
                x_vals = df.iloc[controlsIndex, r]
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

    def correlationMAtrix(self, df):
        None


# class Epoch:
#     def __init__(self,Fs,dt):
#         self._Fs=Fs
#         self._dt=int(dt*self._Fs)
#         self.epochs=dict()

#     def dt(self):
#         print(self._dt)

#     def Fs(self):
#         print(self._Fs)

#     def __subject_trial_phase_epochs(self,subject,trial,phase,data,annotns,offset):
#         epochs=dict()
#         events = np.unique(annotns[subject][trial][phase][1,:]).tolist()
#         no_chs=data[subject][trial][phase].shape[0]
#         for event in events:
#             epochs[event]=np.empty((0,no_chs,self._dt))
#         for e,event in enumerate(annotns[subject][trial][phase][1,:]):
#             t0=int(float(annotns[subject][trial][phase][0,e])*self._Fs-(offset*self._Fs))
#             t1=t0+self._dt
#             dat=data[subject][trial][phase][:,t0:t1]
#             if dat.size==no_chs*self._dt:
#                 epochs[event]=np.vstack((
#                     epochs[event],
#                     data[subject][trial][phase][:,t0:t1].reshape(1,no_chs,self._dt)
#                     ))
#             else:
#                 pass
#         return epochs

#     def subject_phase_epochs(self,subject,phase,data,annotns,offset):
#         no_chs=data[subject][0][phase].shape[0]
#         trials_epochs_list=[]
#         for trial in range(len(annotns[subject])):
#             trials_epochs_list.append(self.__subject_trial_phase_epochs(subject,trial,phase,data,annotns,offset))

#         for i in range(len(trials_epochs_list)-1):
#             assert trials_epochs_list[i].keys() == trials_epochs_list[i+1].keys()
#         ks = trials_epochs_list[0].keys()

#         trials_epochs=dict()
#         for k in ks:
#             trials_epochs[k] = np.empty((0,no_chs,self._dt))
#             for trial in trials_epochs_list:
#                 trials_epochs[k] = np.vstack((
#                     trials_epochs[k],
#                     trial[k]
#                 ))

#         return trials_epochs

#     def all_subject_phase_epochs(self,phase,data,annotns,offset):
#         no_chs=data[0][0][phase].shape[0]
#         subjects_phase_epochs_list=list()
#         for subject in range(len(annotns)):
#             subjects_phase_epochs_list.append(self.subject_phase_epochs(subject,phase,data,annotns,offset))

#         for j in range(len(subjects_phase_epochs_list)-1):
#             assert subjects_phase_epochs_list[j].keys() ==  subjects_phase_epochs_list[j+1].keys()
#         ks = subjects_phase_epochs_list[0].keys()

#         subject_phase_epochs=dict()
#         for k in ks:
#             subject_phase_epochs[k] = np.empty((0,no_chs,self._dt))
#             for subject in subjects_phase_epochs_list:
#                 subject_phase_epochs[k] = np.vstack((
#                     subject_phase_epochs[k],
#                     subject[k]
#                 ))

#         return subject_phase_epochs


#     def average_epochs(self,data):
#         assert isinstance(data, dict)
#         averaged=dict()

#         for k in data.keys():
#             averaged[k] = np.mean(data[k],0)

#         return averaged,averaged.keys()

#     def plot_by_channels(self,t0,t1,offset,data,electrodes,fig_size=(20,10)):
#         assert isinstance(data,dict)
#         if (t0<0) or (t0-offset<0):
#             t0 = int(t0*self._Fs)
#             t1 = int(t1*self._Fs)
#         else:
#             t0 = int((t0-offset)*self._Fs)
#             t1 = int((t1-offset)*self._Fs)
#         offset=int(offset*self._Fs)
#         # no_chs = data[list(data.keys())[0]].shape[0]
#         no_chs = len(electrodes)
#         if no_chs <= 8:
#             n_rows = 2
#         elif no_chs <= 16:
#             n_rows = 4
#         elif no_chs <= 24:
#             n_rows = 6
#         n_cols = int(no_chs/n_rows)
#         fig, ax = plt.subplots(n_rows,n_cols,figsize=fig_size)

#         for r in range(n_rows):
#             for c in range(n_cols):
#                 lgnd=list()
#                 for k in data.keys():
#                     # if k != ' ':
#                     lgnd.append(k)
#                     ax[r,c].plot(data[k][(r*n_cols)+c,t0:t1],'--')
#                     plt.legend(lgnd)
#                     ax[r,c].set_xticks(list(range(0,data[k][(r*n_cols)+c,t0:t1].shape[0],10)))
#                     # ax[r,c].set_xticklabels(list(range(-1*offset,t1-offset+10,10)))
#                     ax[r,c].set_xticklabels(list(range(t0,t1,10)))
#                     ax[r,c].set_title(electrodes[(r*n_cols)+c])
#                     ax[r,c].set_xlabel('ms')
#                     ax[r,c].set_ylabel('uV')

#         fig.tight_layout()
#         plt.show()

#     def time_montage_plot(self,electrodes,t0,t1,offset,data):
#         assert isinstance(data,dict)
#         t0 = int(t0*self._Fs)
#         t1 = int(t1*self._Fs)
#         offset=int(offset*self._Fs)
#         n_rows = len(data)
#         n_cols = t1-t0

#         fig,ax = plt.subplots(n_rows,n_cols)

#         for r,k in enumerate(data.keys()):
#             for t in range(n_cols):
#                 plots.montage_plot(data[0:21,t0+t],electrodes,ax[r,c])
