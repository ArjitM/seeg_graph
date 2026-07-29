"""
Code to derive functional connectivity measures from sEEG raw data.
Implemented for either resting state LFPs or annotated seizure.

Output is saved as .npy (numerical data) and .json (annotations)

args: Input is the singular path of an EDF file to be read.

Key assumptions:
1. EEG +/- EKG channels only. No EOG.
2. All data collected in the US, thus utility frequency is assumed (and validated) to be 60Hz.
   Data is 60Hz notch filtered, and +/- 2 Hz are excluded at 60, 120, 180 Hz for connectivity analysis (see FREQ_BANDS).
   !!! Revisit if using European/Asian datasets !!!
3. Sampling rate > 500 Hz. If lower, gamma3 should likely be excluded (see FREQ_BANDS).
4. Bipolar referencing between adjacent contacts. Missing contacts are "allowed" such that other contiguous contacts
   on that lead are retained. If the desired behavior is to exclude the lead entirely, changes must be made.

Requirements:
Intended use with python3.12
Likely compatible with python3.5+
"""

__author__ = "Arjit Misra"
__email__ = ["arjitm@uchicago.edu", "arjitm2@illinois.edu"]
__version__ = "2026-July-21"

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time, phase_slope_index_time
from mne.preprocessing import ICA
import re
import sys
import shlex
import argparse
import json
import ast
from fooof import FOOOF

UTILITY_FREQ = 60  # all recordings in the US, 60hz utility
METRICS = ['plv', 'wpli'] #'dpli', 'wpli2_debiased', ]# 'dpli']

# Beta split into low and high bands as this may differ in cortico-cortico and cortico-thalamic circuits, in particular
# w.r.t. directed connectivity measures
# Gamma is split following previous work where low vs high differences were appreciated w.r.t. (pre-)ictal states.
# Gamma 2/3 split is largely to exclude 120 +/- 2 Hz harmonic
FREQ_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'beta1': [13, 18],
    'beta2': [18, 30],
    'gamma1': [30, 58],
    'gamma2': [62, 118],
    'gamma3': [122, 150],
}


### ======================= DO NOT EDIT BELOW THIS LINE ======================= ###


def make_bipolar(raw: mne.io.BaseRaw):

    # can handle space or no space prior to contact name, but expects LEAD<CONTACT_NUM> format
    c_id = [(c, c.split(' ')[-1]) for c in raw.copy().pick(picks='eeg').ch_names]

    ch_names, contacts = zip(*c_id)
    ch_name_by_contact = dict({*zip(contacts, ch_names)})
    pattern = re.compile(r"([A-Za-z]+)(\d+)$") #
    contacts_by_lead = {}
    for c in contacts:
        m = pattern.match(c)
        if m:
            c_id = m.group(1)
            contacts_by_lead[c_id] = contacts_by_lead.get(c_id, list())
            contacts_by_lead[c_id].append(int(m.group(2)))

    for lead in list(contacts_by_lead.keys()):  # keep list() expression to avoid write while iterating
        contacts_by_lead[lead] = sorted(contacts_by_lead.get(lead))

    bp_args = {
        'anode': list(),
        'cathode': list(),
        'ch_name': list(),
    }

    for lead, contacts in contacts_by_lead.items():
        '''
        contacts are contiguous iff np.unique(np.diff(contacts))[0] == 1

        example case 
        >>> non_contig = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]  # 6, 10 missing
        then
        >>> cl = np.where(np.diff(non_contig) == 1)[0]
        >>> cl
        array([ 0,  1,  2,  3,  5,  6,  8,  9, 10, 11])
        >>> [non_contig[c] for c in cl]
        [1, 2, 3, 4, 7, 8, 11, 12, 13, 14]'

        Note that 4-5 is OK; 5-6 and 6-7 do not exist. Likewise for 8-9 OK. 9-10, 10-11 do not exist. 
        This is the desired behavior for missing contacts 6 and 10. 
        '''
        contig_locations = np.where(np.diff(contacts) == 1)[0]
        to_use = [contacts[ii] for ii in contig_locations]

        bp_args['anode'] += [ch_name_by_contact.get(f'{lead}{c}') for c in to_use]
        bp_args['cathode'] += [ch_name_by_contact.get(f'{lead}{c + 1}') for c in to_use]
        bp_args['ch_name'] += [f'{lead}_{c}-{c + 1}' for c in to_use]

    return mne.set_bipolar_reference(raw, **bp_args), bp_args.get('ch_name')


def resting_state(raw_epochs: mne.Epochs):

    results_cache = {band: dict() for band in FREQ_BANDS.keys()}

    for band, limits in FREQ_BANDS.items():

        freqs = np.logspace(np.log10(limits[0]), np.log10(limits[1]), num=10)
        con = spectral_connectivity_time(raw_epochs,
                                         n_cycles=8,
                                         freqs=freqs,
                                         method=METRICS,
                                         sfreq=raw_epochs.info.get('sfreq'),
                                         mode='multitaper',
                                         fmin=limits[0], fmax=limits[1],
                                         n_jobs=1,
                                         average=True,
                                         faverage=True)

        eff = phase_slope_index_time(raw_epochs,
                                     freqs=freqs,
                                     fmin=limits[0], fmax=limits[1],
                                     sfreq=raw_epochs.info.get('sfreq'),
                                     mode='cwt_morlet',
                                     average=True,
                                     n_cycles=8)

        for i, result in enumerate(con):  # expect an array of [m x m x 1] matrices, where m = # of contacts
            synch = result.get_data(output='dense')
            synch = synch.squeeze(axis=-1)  # [m x m x 1] -> [m x m]
            # np.save(EDF_INPUT.replace(".edf", f"_synchrony_{band}_{METRICS[i]}.npy"), synch)
            results_cache[band][METRICS[i]] = synch

        results_cache[band]['psi'] = eff.get_data(output='dense').squeeze(axis=-1)

    return  results_cache


def event_based(signal: mne.io.BaseRaw):
    """
    Functional and effective connectivity for one event/time series
    :param signal:
    :return:
    """
    results_cache = {band: dict() for band in FREQ_BANDS.keys()}

    for band, limits in FREQ_BANDS.items():

        freqs = np.logspace(np.log10(limits[0]), np.log10(limits[1]), num=10)

        # process as a single epoch
        con = spectral_connectivity_time(np.array([signal.get_data()]),
                                         method=METRICS, sfreq=signal.info.get('sfreq'),
                                         mode='cwt_morlet',
                                         # decim=20,
                                         freqs=freqs,
                                         fmin=limits[0], fmax=limits[1],
                                         n_cycles=8,
                                         average=True,
                                         faverage=True)

        eff = phase_slope_index_time(np.array([signal.get_data()]),
                                     freqs=freqs,
                                     fmin=limits[0], fmax=limits[1],
                                     sfreq=signal.info.get('sfreq'),
                                     mode='cwt_morlet',
                                     average=True,
                                     n_cycles=8)

        for i, result in enumerate(con):
            synch = result.get_data(output='dense')
            synch = synch.squeeze(axis=-1)  # [m x m x 1] -> [m x m]
            results_cache[band][METRICS[i]] = synch

        results_cache[band]['psi'] = eff.get_data(output='dense').squeeze(axis=-1)

    return results_cache


def time_locked_events(signals):
    """
    Method to characterize a prototypical time-locked event (e.g. seizure) from several individual replicates.
    Functional and effective connectivity averaged across events.
    :param signals: list or numpy array of mne Raw signals OR instance of mne Epochs representing individual seizures
    where t=0 is epileptologist-labeled seizure onset time.
    :return: time-locked, inter-epoch averaged functional/effective connectivity measures
    """
    return



def ictal(raw: mne.io.BaseRaw):

    with open(EDF_INPUT.replace('.edf', '.json'), 'r', encoding='utf-8') as ff:
        annttns = json.load(ff)

    task = annttns.get("TaskDescription", None)

    if task is not None:
        sz = float(re.search(r"sz_onset_time_s:\s*([\d.]+)", task).group(1))

        ioz = ast.literal_eval(
            re.search(r"IOZ:\s*(\[.*\])", task).group(1)
        )

    else:
        return

    ### logic for processing pre-/post-/ictal
    preictal_signal = raw.copy().crop(tmin=0.0, tmax=sz)
    ictal_signal = raw.copy().crop(tmin=sz, reset_first_samp=True)

    RC = event_based(preictal_signal)

    # np.save(EDF_INPUT.replace(".edf", ".npy"), RC)
    with open(EDF_INPUT.replace('.edf', '_connectivity.json'), 'w', encoding='utf-8') as ff:
        json.dump(RC, ff, indent=4)


def run():
    raw = mne.io.read_raw_edf(EDF_INPUT, preload=True)
    raw.filter(l_freq=0.5, h_freq=None)
    raw.set_channel_types({c: 'ecg' if ('ecg' in c.lower() or 'ekg' in c.lower()) else 'eeg' for c in raw.ch_names})
    raw.notch_filter(60, 'eeg')

    ica = ICA(n_components=None, method='fastica', random_state=14)
    ica.fit(raw)
    ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=raw.copy().pick(picks='ecg').ch_names[0], method='correlation')
    ica.exclude = ecg_inds
    ica.apply(raw)

    raw_bipolar, bp_channels = make_bipolar(raw)

    raw_epochs = mne.make_fixed_length_epochs(raw_bipolar, duration=20, overlap=10, verbose=False)

    # resting_state(raw_epochs)
    ictal(raw_bipolar)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("--resting_state", type=bool, default=True)
    parser.add_argument("--json_annotations", type=bool, default=False)

    args = parser.parse_args()

    EDF_INPUT = " ".join(shlex.split(args.input_file))

    if not EDF_INPUT.lower().endswith('.edf'):
        print(EDF_INPUT)
        raise ValueError("needs EDF file input")
    run()

