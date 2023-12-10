import os

import numpy as np
from pyedflib import EdfReader


def make_path(root_path, subject_id=1, trial=1, phase=4):
    return (
        root_path
        + str(subject_id)
        + "/"
        + str(trial)
        + "/"
        + "Phase "
        + str(phase)
        + ".edf"
    )


def xtract_id_subject_info_dict(dir):
    subject_ids = os.listdir(dir)
    id_subject_dict = dict()
    for id in subject_ids:
        with open(dir + id + "/" + "userfile.gnr", "rb") as gnr:
            gnr_content = gnr.read()
            index = list()
            for i, b in enumerate(gnr_content):
                if b == 0x0A:
                    index.append(i)
            name = (
                str(gnr_content[index[0] + 11 : index[1] - 1])[2:-1]
                + " "
                + str(gnr_content[index[1] + 9 : index[2] - 1])[2:-1]
            )
            age = str(gnr_content[index[2] + 5 : index[3] - 1])[2:-1]
            sex = str(gnr_content[index[3] + 5 : index[4] - 1])[2:-1]
            category = str(gnr_content[index[4] + 10 : index[5] - 1])[2:-1]
        id_subject_dict[id] = {
            "name": name,
            "age": int(age),
            "sex": sex,
            "category": category,
        }
        (
            id_subject_dict[id]["eeg_data"],
            id_subject_dict[id]["eeg_markers"],
            id_subject_dict[id]["Fs"],
            id_subject_dict[id]["chs"],
            id_subject_dict[id]["ds"],
        ) = xtract_subject_all_trials(dir, id)

    return id_subject_dict


def xtract_subject_trial(root_path, subject_id, trial):
    if trial != "userfile.gnr":
        data = dict.fromkeys([1, 2, 3, 4])
        annotns = dict.fromkeys([1, 2, 3, 4])
        Fs = {1: 0, 2: 0, 3: 0, 4: 0}
        chs = dict.fromkeys([1, 2, 3, 4])
        Ts = dict.fromkeys([1, 2, 3, 4])
        for phase in data.keys():
            try:
                # open edf recording
                temp = EdfReader(
                    make_path(root_path, int(subject_id), int(trial), int(phase))
                )
                no_chs = temp.signals_in_file

                # check frequency is okay
                Fs[phase] = temp.getSampleFrequencies()[0]
                if Fs[phase] == 100 or Fs[phase] == 0:
                    raise OSError

                # channel_names
                chs[phase] = temp.getSignalLabels()

                # record duration
                Ts[phase] = temp.datarecord_duration

                # fetch EEG data
                assert np.array_equal(temp.getNSamples(), temp.getNSamples())
                N = temp.getNSamples()[0]
                data[phase] = np.empty((0, N))
                for ch in range(no_chs):
                    data[phase] = np.vstack((data[phase], temp.readSignal(ch)))
                temp_exist = True

                # fetch markers
                annot = temp.readAnnotations()
                annotns[phase] = np.vstack((annot[0], annot[2]))
                temp_exist = True
            except OSError:
                temp_exist = False
                data[phase] = np.empty((0, 0, 0))
                annotns[phase] = np.empty((0, 0, 0))
        if temp_exist == True:
            temp.close()
        return (data, annotns, Fs, chs, Ts)
    else:
        return None


def xtract_subject_all_trials(root_path, subject_id):
    all_trials = {
        "rest1": list(),
        "arith": list(),
        "rest2": list(),
        "auditory": list(),
    }
    all_markers = {
        "rest1": list(),
        "arith": list(),
        "rest2": list(),
        "auditory": list(),
    }
    fs = {
        "rest1": list(),
        "arith": list(),
        "rest2": list(),
        "auditory": list(),
    }
    ch_s = {
        "rest1": list(),
        "arith": list(),
        "rest2": list(),
        "auditory": list(),
    }
    ds = {
        "rest1": list(),
        "arith": list(),
        "rest2": list(),
        "auditory": list(),
    }
    for trial in os.listdir(root_path + "/" + str(subject_id)):
        if trial != "userfile.gnr":
            trial, marker, Fs, chs, Ds = xtract_subject_trial(
                root_path, subject_id, int(trial)
            )
            all_trials["rest1"].append(trial[1])
            all_trials["arith"].append(trial[2])
            all_trials["rest2"].append(trial[3])
            all_trials["auditory"].append(trial[4])

            all_markers["rest1"].append(marker[1])
            all_markers["arith"].append(marker[2])
            all_markers["rest2"].append(marker[3])
            all_markers["auditory"].append(marker[4])

            fs["rest1"].append(Fs[1])
            fs["arith"].append(Fs[2])
            fs["rest2"].append(Fs[3])
            fs["auditory"].append(Fs[4])

            ch_s["rest1"].append(chs[1])
            ch_s["arith"].append(chs[2])
            ch_s["rest2"].append(chs[3])
            ch_s["auditory"].append(chs[4])

            ds["rest1"].append(Ds[1])
            ds["arith"].append(Ds[2])
            ds["rest2"].append(Ds[3])
            ds["auditory"].append(Ds[4])

    return all_trials, all_markers, fs, ch_s, ds
