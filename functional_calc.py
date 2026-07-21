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
from mne_connectivity import spectral_connectivity_epochs
from mne.preprocessing import ICA
import re
import sys
import shlex
from fooof import FOOOF

UTILITY_FREQ = 60  # all recordings in the US, 60hz utility
METRICS = ['plv', 'wpli2_debiased', 'dpli']

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
    'gamma3': [122, 178],
}


### ======================= DO NOT EDIT BELOW THIS LINE ======================= ###


def make_bipolar(raw):

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


def resting_state(raw_epochs):

    for band, limits in FREQ_BANDS.items():

        con = spectral_connectivity_epochs(raw_epochs,
                                         #n_cycles=4, freqs=freqs,
                                         method=METRICS, sfreq=raw_epochs.info.get('sfreq'),
                                         mode='multitaper',
                                         fmin=limits[0], fmax=limits[1],
                                         n_jobs=1,
                                         faverage=True)

        for i, xx in enumerate(con):  # expect an array of matrices, each [m x m x 1] where m is number of contacts.
            synch = xx.get_data(output='dense')
            synch = synch.reshape(synch.shape[:2])  # [m x m x 1] -> [m x m]
            np.save(EDF_INPUT.replace(".edf", f"_synchrony_{band}_{METRICS[i]}.npy"), synch)



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

    resting_state(raw_epochs)




if __name__ == '__main__':
    xx = sys.argv[1]
    print(xx)
    EDF_INPUT = " ".join(shlex.split(sys.argv[1]))
    if not EDF_INPUT.lower().endswith('.edf'):
        print(EDF_INPUT)
        raise ValueError("needs EDF file input")
    run()

