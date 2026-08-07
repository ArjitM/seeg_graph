"""
Code to derive functional connectivity measures from sEEG raw data.
Implemented for either resting state LFPs or annotated seizure.

Output is saved as .pkl.gz (gunzip compressed pickle).

args:
INPUT_FILE is the singular path of an EDF file to be read.
RESTING_STATE (optional; default True) whether signal is resting state. If False, presumed to be a seizure recording
JSON_ANNOTATIONS (optional; default False) whether a JSON file of the same name as INPUT_FILE contains annotations.
OUTPUT_DIR (optional) where to save output, defaults to directory where INPUT_FILE is located
BATCH_EVENTS (optional; default False) set as TRUE iff you want multiple seizures to be processed, resulting in
inter-event averaged connectivity measures to characterize the prototypical event. In this case, INPUT_FILE is assumed
to [some prefix]seizure[number].edf; the number is effectively ignored and all files within the parent directory
are parsed if they match the name format.

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
mne v. 1.12
mne_connectivity v. 0.81
Likely compatible with python3.10+
"""

__author__ = "Arjit Misra"
__email__ = ["arjitm@uchicago.edu", "arjitm2@illinois.edu"]
__version__ = "2026-Aug-5"

import os

import numpy as np
import mne
import pandas as pd
import scipy.signal
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time, phase_slope_index_time, phase_slope_index
from mne.preprocessing import ICA
import re
import shlex
import argparse
import json
import ast
import pickle
import gzip
from fooof import FOOOF
from pathlib import Path
from helper_methods import getBipolarChannels

UTILITY_FREQ = 60  # all recordings in the US, 60hz utility
METRICS = ['plv', 'wpli'] #'dpli', 'wpli2_debiased', ]# 'dpli']
METRICS_INTER = ['plv', 'ppc', 'wpli', 'dpli']

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


T_EARLY_ICTAL_END = 10 #sec
T_LATE_ICTAL_END = 30 #sec

### ======================= DO NOT EDIT BELOW THIS LINE ======================= ###


def make_bipolar(raw: mne.io.BaseRaw):

    bp_args = getBipolarChannels(raw.copy().pick(picks='eeg').ch_names)
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

    fname = Path(EDF_INPUT).name.replace('.edf', '')
    with gzip.open(args.output_dir.joinpath(f'{fname}_preictal_connectivity.pkl.gz'), 'wb') as ff:
        pickle.dump(results_cache, ff)


def _event_based_connectivity(signal: mne.io.BaseRaw):
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


def _inter_event(signals: np.ndarray, fs):

    print(signals.shape)
    results_cache = {band: dict() for band in FREQ_BANDS.keys()}
    for band, limits in FREQ_BANDS.items():

        con = spectral_connectivity_epochs(signals,
                                           mode='multitaper',
                                           method=METRICS_INTER,
                                           sfreq=fs,
                                           fmin=limits[0], fmax=limits[1],
                                           faverage=True
                                           )

        eff = phase_slope_index(signals,
                                fmin=limits[0], fmax=limits[1],
                                sfreq=fs,
                                mode='multitaper',
                                )

        for i, result in enumerate(con):
            results_cache[band][METRICS_INTER[i]] = result.get_data(output='dense').squeeze(axis=-1)
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
    time_series = []
    fs_arr = []
    for sz in signals:
        time_series.append(sz.get_data())
        fs_arr.append(sz.info.get('sfreq'))

    fs = np.min(fs_arr)

    if np.unique(fs_arr).shape[0] != 1:  # check if different sampling rates
        resampled = []
        for i, ts in enumerate(time_series):
            resampled.append(scipy.signal.resample(ts, int(len(ts) * fs / fs_arr[i])))
    else:
        resampled = time_series


    early_ictal_stack = np.array([
        rs[:, : int(fs * T_EARLY_ICTAL_END)]
        for rs in resampled
    ])

    res = _inter_event(early_ictal_stack, fs)

    fname = Path(EDF_INPUT).name.split('seizure')[0]
    with gzip.open(args.output_dir.joinpath(f'{fname}_early_ictal_prototype.pkl.gz'),'wb') as ff:
        pickle.dump(res, ff)

    late_ictal_stack = np.array([
            rs[:, int(fs * T_EARLY_ICTAL_END) : int(fs * T_LATE_ICTAL_END) ]
            for rs in resampled
        ])

    res = _inter_event(late_ictal_stack, fs)

    with gzip.open(args.output_dir.joinpath(f'{fname}_late_ictal_prototype.pkl.gz'), 'wb') as ff:
        pickle.dump(res, ff)


def _ictal_preictal_split(raw, sig_file):

    with open(sig_file.replace('.edf', '.json'), 'r', encoding='utf-8') as ff:
        annttns = json.load(ff)

    task = annttns.get("TaskDescription", None)

    if task is not None:
        sz = float(re.search(r"sz_onset_time_s:\s*([\d.]+)", task).group(1))

        ioz = ast.literal_eval(
            re.search(r"IOZ:\s*(\[.*\])", task).group(1)
        )

    else:
        return

    preictal_signal = raw.copy().crop(tmin=0.0, tmax=sz)
    ictal_signal = raw.copy().crop(tmin=sz, reset_first_samp=True)

    return {
        "pre-ictal": preictal_signal,
        "ictal": ictal_signal,
        "channels": ioz,
    }


def ictal(raw: mne.io.BaseRaw):

    sz_data = _ictal_preictal_split(raw, EDF_INPUT)

    preictal_signal = sz_data.get('pre-ictal')
    ictal_signal = sz_data.get('ictal')

    conn = _event_based_connectivity(preictal_signal)

    fname = Path(EDF_INPUT).name.replace('.edf', '')
    with gzip.open(args.output_dir.joinpath(f'{fname}_preictal_connectivity.pkl.gz'),'wb') as ff:
        pickle.dump(conn, ff)

    conn = _event_based_connectivity(ictal_signal)

    with gzip.open(args.output_dir.joinpath(f'{fname}_ictal_connectivity.pkl.gz'),'wb') as ff:
        pickle.dump(conn, ff)


def _preprocess_to_bipolar(raw):
    raw.filter(l_freq=0.5, h_freq=None)
    raw.set_channel_types({c: 'ecg' if ('ecg' in c.lower() or 'ekg' in c.lower()) else 'eeg' for c in raw.ch_names})
    raw.notch_filter(UTILITY_FREQ, 'eeg')

    ica = ICA(n_components=None, method='fastica', random_state=14)
    ica.fit(raw)
    ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=raw.copy().pick(picks='ecg').ch_names[0], method='correlation')
    ica.exclude = ecg_inds
    ica.apply(raw)
    return make_bipolar(raw)


def _aperiodic_exp(signal1d, fs, f_min, f_max):
    f, pxx = scipy.signal.welch(signal1d, fs, nperseg=256, noverlap=128, detrend='constant')
    m = (f >= f_min) & (f <= f_max) & np.isfinite(pxx) & (pxx > 0)
    if np.sum(m) < 10:
        return np.nan

    fm = FOOOF(peak_width_limits=(1, 12), max_n_peaks=6, verbose=False)
    fm.fit(f[m], pxx[m])
    ap = fm.get_params("aperiodic_params")  # [offset, exponent]
    return float(ap[1])


def _bandpower(signal1d, fs):

    nperseg = min(len(signal1d), int(fs * min(10, int(len(signal1d) / fs))))
    f, psd = scipy.signal.welch(signal1d, fs=fs, nperseg=nperseg)
    band_powers = {}
    for b, f_lim in FREQ_BANDS.items():
        # numerical integration over frequency range
        band_powers[f'{b}_power'] = np.trapz(
            psd[(f >= f_lim[0]) & (f < f_lim[1])],
            x=f[(f >= f_lim[0]) & (f < f_lim[1])]
        )
    return band_powers


def _entropy(signal1d, fs):

    dur_norm = len(signal1d) * fs / 1000  # relative length of signal, for uniformity in binning
    num_bins = int(dur_norm ** 0.5)

    # Discretize the signal into bins, then normalize
    histogram, _ = np.histogram(signal1d, bins=num_bins, density=True)
    probabilities = histogram / np.sum(histogram)
    probabilities = probabilities[probabilities > 0]

    return scipy.stats.entropy(probabilities)  # shannon entropy


def _node_features(signal2d: np.ndarray, fs, ch_names) -> pd.DataFrame:
    ap_exp, band_pow, entropy = list(), list(), list()
    for sig_ch in signal2d:
        ap_exp.append(_aperiodic_exp(sig_ch, fs, f_min=5, f_max=55))
        band_pow.append(_bandpower(sig_ch, fs))  # list of dicts
        entropy.append(_entropy(sig_ch, fs))

    df = pd.concat([
        pd.DataFrame.from_dict({'ch_name': ch_names,
                                'aperiodic_exponent': ap_exp,
                                'entropy': entropy,}),
        pd.DataFrame(band_pow),
    ], axis=1)
    return df


def single_channel(raw: mne.io.BaseRaw | mne.Epochs, dynamic: bool):
    """
    Derive single-channel waveform and spectral features, aperiodic exponent.
    Saves CSV file in OUTPUT_DIR
    """
    fs = raw.info.get('sfreq')
    fname = Path(EDF_INPUT).name.replace('.edf', '')


    if dynamic:
        sz_data = _ictal_preictal_split(raw, EDF_INPUT)
        preictal_signal = sz_data.get('pre-ictal')
        ictal_signal = sz_data.get('ictal')
        pdf = _node_features(preictal_signal, fs, raw.ch_names)
        pdf.to_csv(args.output_dir.joinpath(f'{fname}_preictal_node_level.csv'))
        idf = _node_features(ictal_signal, fs, raw.ch_names)
        idf.to_csv(args.output_dir.joinpath(f'{fname}_ictal_node_level.csv'))

    else:
        signal2d_epochs = raw.get_data()
        epoch_dfs = [_node_features(s2d, fs, raw.ch_names) for s2d in signal2d_epochs]
        df = pd.concat(epoch_dfs).groupby(level=0).mean()
        df.to_csv(args.output_dir.joinpath(f'{fname}_resting_node_level.csv'))


def run():

    if args.batch_events:
        # Batch process seizures as time-locked events and derive metrics across events
        parent_dir = Path(EDF_INPUT).parent
        signals = []
        channels = []
        for ff in os.listdir(parent_dir):
            if ff.endswith('.edf') and 'seizure' in ff.lower():
                raw_sig = mne.io.read_raw_edf(parent_dir.joinpath(ff), preload=True)
                raw_bp, _ = _preprocess_to_bipolar(raw_sig)
                sz_data = _ictal_preictal_split(raw_bp, str(parent_dir.joinpath(ff)))
                signals.append(sz_data.get('ictal'))
                channels.append(sz_data.get('channels'))
        time_locked_events(signals, channels, parent_dir)
        return

    raw = mne.io.read_raw_edf(EDF_INPUT, preload=True)
    raw_bipolar, bp_channels = _preprocess_to_bipolar(raw)

    if args.resting_state:
        # For resting state signal, split into overlapping time epochs and average over them
        raw_epochs = mne.make_fixed_length_epochs(raw_bipolar, duration=20, overlap=10, verbose=False)
        resting_state(raw_epochs)

    else:
        # Each ictal recording processed individually. For patient-level batch processing use BATCH_EVENTS=True.
        # Note recordings contain pre-ictal signal. Time series is split as per annotations.
        ictal(raw_bipolar)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("--resting_state", type=bool, default=True)
    parser.add_argument("--json_annotations", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_events", type=bool, default=False)

    args = parser.parse_args()

    EDF_INPUT = " ".join(shlex.split(args.input_file))  # works with output of <ls> including /path\ with\ spaces/

    if args.output_dir is not None:
        args.output_dir = Path(" ".join(shlex.split(args.output_dir)))
    else:
        args.output_dir = Path(EDF_INPUT).parent  # if output_dir not explicitly provided, use input_file directory

    if not EDF_INPUT.lower().endswith('.edf'):
        print(EDF_INPUT)
        raise ValueError("needs EDF file input")
    run()

