"""
Functional connectivity, node features, and recruitment from bipolar sEEG.

One seizure per EDF. JSON sidecar: sz_onset_time_s and SOZ (or IOZ). Optional
sz_offset_time_s is measured from EDF start. SOZ names may be bipolar names
or monopolar endpoints; either endpoint marks a bipolar channel as SOZ.

Seizure states: fixed preictal baseline, early ictal [0, 10), later ictal
[10, 30) seconds. Each state uses complete, nonoverlapping 10-second windows
by default. Short states are reported and skipped, never padded or mixed
with postictal data. Recruitment uses the continuous recording separately.

--interictal_file adds a separate EDF once per run. --resting_state remains
available for interictal-only processing. --batch_events discovers matching
seizure-prefix files; --seizure_files explicitly supplies additional files.
Batch mode retains individual-seizure results; --prototype additionally
requests the original across-seizure estimator (different estimand).

Outputs: per-state connectivity .pkl.gz, per-window and averaged node CSVs,
recruitment CSV, and per-recording metadata JSON. Connectivity payloads have
keys metadata and connectivity; access payload['connectivity'][band][metric].

Key assumptions:
1. EEG +/- EKG channels only. No EOG. Explicit bad channels can be dropped.
2. All recordings in the US: 60-Hz notch; frequency bands avoid 60/120 Hz.
3. Bipolar pairs come from your existing helper_methods.getBipolarChannels.
4. Channels must match within an implantation; files are reordered by name.
   Use separate runs for separate implantations or different channel sets.
5. No automatic artifact rejection beyond optional ECG ICA. Review data first.

Requires Python 3.10+, MNE 1.12, mne-connectivity 0.8.1+, fooof,
numpy, scipy, pandas, helper_methods.py, and sz_spread.py.
"""
__author__ = "Arjit Misra"
__email__ = ["arjitm@uchicago.edu", "arjitm2@illinois.edu"]
__version__ = "2026-Sep-30"

import warnings
import datetime
import numpy as np
import mne
import pandas as pd
import scipy.signal
import scipy.integrate
import scipy.stats
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time, phase_slope_index_time, phase_slope_index
from mne.preprocessing import ICA
import re
import argparse
import json
import ast
import pickle
import gzip
from fooof import FOOOF
from pathlib import Path
from helper_methods import getBipolarChannels
from sz_spread import detect_recruitment

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


def make_bipolar(raw: mne.io.BaseRaw, exclude_pairs):
    bp_args = getBipolarChannels(raw.copy().pick(picks='eeg').ch_names, exclude_pairs=exclude_pairs)
    names = list(bp_args['ch_name'])
    if not names or len(names) != len(set(names)):
        raise ValueError('Bipolar names must be nonempty and unique.')
    bp_referenced = mne.set_bipolar_reference(raw, **bp_args)
    bp_referenced.pick(names)  # Preserve helper order; never use set() here.
    return bp_referenced, names


def _preprocess_to_bipolar(raw, exclude_pairs, skip_ica=False):
    raw.set_channel_types({c: 'ecg' if ('ecg' in c.lower() or 'ekg' in c.lower()) else 'eeg'
                           for c in raw.ch_names})
    raw.filter(l_freq=0.5, h_freq=None, picks='eeg')
    if UTILITY_FREQ >= raw.info['sfreq'] / 2:
        raise ValueError('Sampling rate must permit the 60-Hz notch.')
    raw.notch_filter(UTILITY_FREQ, picks='eeg')
    ecg = raw.copy().pick(picks='ecg').ch_names if 'ecg' in raw.get_channel_types() else []
    if not skip_ica and ecg:
        ica = ICA(n_components=None, method='fastica', random_state=14)
        ica.fit(raw, picks='eeg')
        ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=ecg[0], method='correlation')
        ica.exclude = ecg_inds
        ica.apply(raw)
    elif not skip_ica:
        warnings.warn('No ECG channel: skipping ECG ICA.')
    return make_bipolar(raw, exclude_pairs)


def _align_channels(raw, names):
    missing, extra = set(names) - set(raw.ch_names), set(raw.ch_names) - set(names)
    if missing or extra:
        raise ValueError(f'Channel mismatch: missing={sorted(missing)}, extra={sorted(extra)}. '
                         'Choose a reviewed common channel set before running this comparison.')
    return raw.reorder_channels(names)


def _read_annotations(sig_file):
    with open(Path(sig_file).with_suffix('.json'), encoding='utf-8') as ff:
        annotations = json.load(ff)
    task = annotations.get('TaskDescription') or ''

    def number(key):
        value = annotations.get(key)
        if value is None:
            match = re.search(rf'{key}:\s*([\d.]+)', task)
            value = match.group(1) if match else None
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'{key} must be finite.')
        return value

    soz = annotations.get('SOZ', annotations.get('IOZ'))
    if soz is None:
        match = re.search(r'(?:SOZ|IOZ):\s*(\[[^\]]*\])', task)
        soz = ast.literal_eval(match.group(1)) if match else []
    if not isinstance(soz, list) or not all(isinstance(ch, str) for ch in soz):
        raise ValueError('SOZ/IOZ must be a list of channel-name strings.')
    return {'onset_s': number('sz_onset_time_s'),
            'onset_datetime': annotations.get('sz_onset_datetime', None),
            # 'offset_s': number('sz_offset_time_s'),
            'soz': soz,
            'bad_channels': annotations.get('bad_channels', list()),}


def _crop_samples(raw, start_s, stop_s):
    """Half-open [start, stop) interval, rounded to the nearest sample."""
    fs = raw.info['sfreq']
    start, stop = int(round(start_s * fs)), int(round(stop_s * fs))
    if not 0 <= start < stop <= raw.n_times:
        raise ValueError(f'Invalid interval [{start_s}, {stop_s}) for {raw.n_times / fs:g}-s EDF.')
    return raw.copy().crop(tmin=start / fs, tmax=(stop - 1) / fs)


def _ictal_preictal_split(raw, sig_file, preictal_s=30.0):
    annotation = _read_annotations(sig_file)
    sz = annotation.get('onset_s', None)
    sz_time = annotation.get('onset_datetime', None)
    if sz is None:
        if sz_time is None:
            raise ValueError("Seizure onset time not provided")
        recording_start = raw.info.get('meas_date')
        sz_time = datetime.datetime.fromisoformat(sz_time)
        if sz_time.tzinfo is None and recording_start.tzinfo is not None:
            sz_time = sz_time.replace(tzinfo=recording_start.tzinfo)
        sz = (sz_time - recording_start).total_seconds()

    offset = annotation.get('offset_s', None)
    duration = raw.n_times / raw.info['sfreq']
    if not 0 < sz < duration:
        raise ValueError('Seizure onset must be inside the EDF with preictal data available.')
    if offset is not None and not sz < offset <= duration:
        raise ValueError('Seizure offset must follow onset and lie within the EDF.')
    end = min(duration, offset if offset is not None else duration, sz + T_LATE_ICTAL_END)
    bounds = {'preictal': (max(0.0, sz - preictal_s), sz),
              'early_ictal': (sz, min(sz + T_EARLY_ICTAL_END, end)),
              'late_ictal': (sz + T_EARLY_ICTAL_END, end)}
    states = {state: _crop_samples(raw, start, stop) for state, (start, stop) in bounds.items()
              if stop > start}
    return {'states': states, 'bounds': bounds, 'annotation': annotation,
            'ictal': _crop_samples(raw, sz, end)}


def _resolve_soz(soz, bp_args, ch_names):
    """A bipolar channel is SOZ if explicitly labeled or either endpoint is labeled."""
    pairs = dict(zip(bp_args['ch_name'], zip(bp_args['anode'], bp_args['cathode'])))
    valid = set(ch_names) | {endpoint for pair in pairs.values() for endpoint in pair}
    missing = set(soz) - valid
    if missing:
        raise ValueError(f'SOZ labels do not match retained contacts: {sorted(missing)}')
    return [ch for ch in ch_names if ch in soz or any(endpoint in soz for endpoint in pairs[ch])]


def _fixed_epochs(raw, window_s=10.0):
    # Nonoverlapping, equally sized windows for every state. Drop trailing remainder.
    fs = raw.info['sfreq']
    n = int(round(window_s * fs))
    count = raw.n_times // n
    if count == 0:
        raise ValueError(f'Segment is shorter than one {window_s:g}-s window.')
    return np.stack([raw.get_data(start=i * n, stop=(i + 1) * n) for i in range(count)])


def _event_based_connectivity(signal, fs=None, n_cycles=4.0, bands=None, mode='multitaper'):
    """Within-window connectivity, then mean across equal-length windows.

    Explicit pair indices avoid interpreting MNE's uncomputed matrix triangle
    as zero connectivity. PLV/wPLI are symmetric; PSI is antisymmetric.
    Multitaper is shared by every state: 4 cycles at 0.5 Hz fit in 10 seconds.
    """
    if isinstance(signal, mne.io.BaseRaw):
        fs, signal = signal.info['sfreq'], signal.get_data()[None, ...]
    signal = np.asarray(signal)
    pairs = np.triu_indices(signal.shape[1], k=1)
    if len(pairs[0]) == 0:
        raise ValueError('Connectivity requires at least two bipolar channels.')
    results_cache = {}
    for band in bands or FREQ_BANDS:
        limits = FREQ_BANDS[band]
        if limits[1] >= fs / 2:
            raise ValueError(f'{band}: upper frequency must be below Nyquist.')
        if n_cycles / limits[0] > signal.shape[-1] / fs:
            raise ValueError(f'{band}: window is too short for {n_cycles:g} cycles at {limits[0]} Hz.')
        freqs = np.geomspace(*limits, num=10)
        kwargs = dict(freqs=freqs, indices=pairs, sfreq=fs, mode=mode,
                      fmin=limits[0], fmax=limits[1], n_cycles=n_cycles, average=True, n_jobs=1)
        con = spectral_connectivity_time(signal, method=METRICS, faverage=True, **kwargs)
        eff = phase_slope_index_time(signal, **kwargs)
        results_cache[band] = {}
        for method, result in zip([*METRICS, 'psi'], [*con, eff]):
            values = result.get_data().reshape(len(pairs[0]), -1)[:, 0]
            matrix = np.zeros((signal.shape[1], signal.shape[1]))
            matrix[pairs] = values
            matrix[(pairs[1], pairs[0])] = -values if method == 'psi' else values
            results_cache[band][method] = matrix
    return results_cache


def _inter_event_connectivity(signals: np.ndarray, fs, bands=None):
    """Optional legacy across-seizure estimator, not the mean of per-seizure graphs."""
    results_cache = {}
    pairs = np.triu_indices(signals.shape[1], k=1)
    for band in bands or FREQ_BANDS:
        limits = FREQ_BANDS[band]
        con = spectral_connectivity_epochs(signals, mode='multitaper', method=METRICS_INTER,
                                           indices=pairs, sfreq=fs, fmin=limits[0],
                                           fmax=limits[1], faverage=True)
        eff = phase_slope_index(signals, indices=pairs, fmin=limits[0], fmax=limits[1],
                                sfreq=fs, mode='multitaper')
        results_cache[band] = {}
        for method, result in zip([*METRICS_INTER, 'psi'], [*con, eff]):
            values = result.get_data().reshape(len(pairs[0]), -1)[:, 0]
            matrix = np.zeros((signals.shape[1], signals.shape[1]))
            matrix[pairs] = values
            matrix[(pairs[1], pairs[0])] = (-values if method == 'psi' else
                                           1 - values if method == 'dpli' else values)
            results_cache[band][method] = matrix
    return results_cache


def _save_connectivity(path, connectivity, metadata):
    with gzip.open(path, 'wb') as ff:
        pickle.dump({'metadata': metadata, 'connectivity': connectivity}, ff)


def time_locked_events(signals, output_dir, prefix, bands=None, source_files=None):
    """Optional prototypical event from onset-aligned Raw recordings."""
    if len(signals) < 2:
        raise ValueError('Prototype estimation requires at least two seizures.')
    fs = min(sz.info['sfreq'] for sz in signals)
    names = signals[0].ch_names
    aligned = []
    for sz in signals:
        sz = _align_channels(sz.copy(), names)
        if sz.info['sfreq'] != fs:
            sz.resample(fs)  # Resample time, not the channel axis.
        if sz.n_times < int(round(fs * T_LATE_ICTAL_END)):
            raise ValueError('Prototype estimation requires 30 ictal seconds in every seizure.')
        aligned.append(sz.get_data())
    for state, start, stop in [('early_ictal', 0, 10), ('late_ictal', 10, 30)]:
        stack = np.stack([ts[:, round(start * fs):round(stop * fs)] for ts in aligned])
        metadata = {'ch_names': names, 'sfreq': fs, 'state': state,
                    'source_files': source_files, 'estimator': 'across_seizure_multitaper',
                    'n_seizures': len(signals), 'interval_from_onset_s': [start, stop]}
        _save_connectivity(output_dir / f'{prefix}_{state}_prototype.pkl.gz',
                           _inter_event_connectivity(stack, fs, bands), metadata)


def _aperiodic_fit(signal1d, fs, f_min, f_max):
    # Fixed 2-second Welch subwindows: resolution is 0.5 Hz at any sampling rate.
    nperseg = min(len(signal1d), int(round(2 * fs)))
    f, pxx = scipy.signal.welch(signal1d, fs, nperseg=nperseg, noverlap=nperseg // 2,
                               detrend='constant')
    mask = (f >= f_min) & (f <= f_max)
    if mask.sum() < 10 or not np.all(np.isfinite(pxx[mask]) & (pxx[mask] > 0)):
        return np.nan, np.nan, np.nan
    fm = FOOOF(peak_width_limits=(1, 12), max_n_peaks=6, verbose=False)
    fm.fit(f[mask], pxx[mask])
    return float(fm.get_params('aperiodic_params')[1]), float(fm.r_squared_), float(fm.error_)


def _aperiodic_exp(signal1d, fs, f_min, f_max):
    return _aperiodic_fit(signal1d, fs, f_min, f_max)[0]


def _bandpower(signal1d, fs):
    nperseg = min(len(signal1d), int(round(fs * 2)))
    f, psd = scipy.signal.welch(signal1d, fs=fs, nperseg=nperseg)
    band_powers = {}
    for band, (low, high) in FREQ_BANDS.items():
        mask = (f >= low) & (f < high)
        band_powers[f'{band}_power'] = (scipy.integrate.trapezoid(psd[mask], x=f[mask])
                                       if high < fs / 2 and mask.sum() >= 2 else np.nan)
    return band_powers


def _entropy(signal1d, fs, bins=64):
    # Amplitude-histogram entropy in nats; not temporal entropy or spectral entropy.
    counts, _ = np.histogram(signal1d, bins=bins)
    return scipy.stats.entropy(counts[counts > 0])


def _node_features(signal2d: np.ndarray, fs, ch_names, ap_range=(5.0, 55.0)) -> pd.DataFrame:
    rows = []
    for ch, sig_ch in zip(ch_names, signal2d):
        exponent, r2, error = _aperiodic_fit(sig_ch, fs, *ap_range)
        rows.append({'ch_name': ch, 'aperiodic_exponent': exponent, 'aperiodic_r_squared': r2,
                     'aperiodic_error': error, 'entropy': _entropy(sig_ch, fs),
                     **_bandpower(sig_ch, fs)})
    return pd.DataFrame(rows)


def single_channel(epochs, fs, ch_names, ap_range=(5.0, 55.0)):
    """Retain per-window results and average numerical features by channel name."""
    frames = []
    for index, signal2d in enumerate(epochs):
        frame = _node_features(signal2d, fs, ch_names, ap_range)
        frame.insert(0, 'window', index)
        frames.append(frame)
    windows = pd.concat(frames, ignore_index=True)
    summary = windows.drop(columns='window').groupby('ch_name', sort=False).mean(numeric_only=True).reset_index()
    return windows, summary


def _analyze_state(raw, state, source, output_dir, options, start_s=0.0, onset_s=None):
    epochs = _fixed_epochs(raw, options.window_s)
    fs = raw.info['sfreq']
    metadata = {'source_file': str(source), 'patient_id': options.patient_id, 'state': state,
                'ch_names': raw.ch_names, 'sfreq': fs, 'window_s': epochs.shape[-1] / fs,
                'n_windows': len(epochs), 'segment_start_s': start_s, 'onset_s': onset_s,
                'analyzed_duration_s': epochs.shape[0] * epochs.shape[-1] / fs,
                'discarded_tail_s': (raw.n_times - epochs.shape[0] * epochs.shape[-1]) / fs,
                'mode': 'multitaper', 'n_cycles': options.n_cycles, 'bands': options.bands,
                'ap_range': options.ap_range, 'power_units': 'V^2', 'entropy_bins': 64,
                'matrix_diagonal': 0, 'psi_convention': 'positive [i,j]: i leads j',
                'edge_exclusions': 'none; adjacent/shared-contact edges retained'}
    stem = f'{Path(source).stem}_{state}'
    windows, summary = single_channel(epochs, fs, raw.ch_names, tuple(options.ap_range))
    windows['window_start_s'] = start_s + windows['window'] * metadata['window_s']
    windows['window_midpoint_s'] = windows['window_start_s'] + metadata['window_s'] / 2
    for frame in (windows, summary):
        frame.insert(0, 'state', state)
        frame.insert(0, 'recording', Path(source).stem)
        frame.insert(0, 'patient_id', options.patient_id)
    windows.to_csv(output_dir / f'{stem}_node_windows.csv', index=False)
    summary.to_csv(output_dir / f'{stem}_node_level.csv', index=False)
    if not options.skip_connectivity:
        conn = _event_based_connectivity(epochs, fs, options.n_cycles, options.bands)
        _save_connectivity(output_dir / f'{stem}_connectivity.pkl.gz', conn, metadata)
    return metadata


def resting_state(raw, source, output_dir, options):
    # A separate interictal file uses the same estimator and window length as seizures.
    return _analyze_state(raw, 'interictal', source, output_dir, options)


def ictal(raw, source, output_dir, options, bp_args):
    sz_data = _ictal_preictal_split(raw, source, options.preictal_s)
    annotation = sz_data['annotation']
    soz = _resolve_soz(annotation['soz'], bp_args, raw.ch_names)
    if not options.skip_recruitment and not soz:
        raise ValueError(f'{source}: no SOZ labels. Supply them or use --skip_recruitment.')
    metadata = {'source_file': str(source), 'annotation': annotation, 'bipolar_soz': soz,
                'ch_names': raw.ch_names, 'bipolar_pairs': bp_args, 'states': {},
                'offset_unknown': annotation.get('offset_s', None) is None}
    for state, (start, stop) in sz_data['bounds'].items():
        segment = sz_data['states'].get(state)
        if segment is None or segment.n_times < round(options.window_s * raw.info['sfreq']):
            warnings.warn(f'{source}: skipping {state}; shorter than one complete window.')
            metadata['states'][state] = {'status': 'insufficient_duration', 'bounds_s': [start, stop]}
            continue
        metadata['states'][state] = _analyze_state(segment, state, source, output_dir, options,
                                                  start_s=start, onset_s=annotation['onset_s'])
    if not options.skip_recruitment:
        onset = annotation['onset_s']
        end = min(raw.n_times / raw.info['sfreq'],
                  annotation['offset_s'] if annotation['offset_s'] is not None else np.inf,
                  onset + options.recruitment_duration if options.recruitment_duration else np.inf)
        baseline = sz_data['bounds']['preictal']
        # Give the detector the continuous signal, not the onset-cropped ictal array.
        result = detect_recruitment(raw.get_data(), raw.ch_names, raw.info['sfreq'], soz,
                                    measure=options.recruitment_band, baseline_s=baseline,
                                    onset_s=onset, search_end_s=end)
        result.insert(0, 'recording', Path(source).stem)
        result.insert(0, 'patient_id', options.patient_id)
        result.to_csv(output_dir / f'{Path(source).stem}_recruitment.csv', index=False)
        metadata['recruitment'] = {'baseline_s': baseline, 'search_end_s': end,
                                  'reference': 'annotated_onset', 'measure': options.recruitment_band,
                                  'window_s': 1.0, 'step_s': 0.25, 'threshold_z': 3.0,
                                  'min_duration_s': 2.0, 'max_gap_s': 0.5}
    return metadata, sz_data['ictal']


def run(options):
    source = Path(options.input_file).expanduser()
    output_dir = Path(options.output_dir).expanduser() if options.output_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [source]
    if options.batch_events:
        match = re.search(r'seizure', source.stem, re.I)
        if match is None:
            raise ValueError('--batch_events needs a filename containing seizure; use --seizure_files otherwise.')
        prefix = source.stem[:match.start()]
        pattern = re.compile(re.escape(prefix) + r'seizure.*', re.I)
        files = sorted(p for p in source.parent.iterdir()
                       if p.suffix.lower() == '.edf' and pattern.fullmatch(p.stem))
    files += [Path(p).expanduser() for p in options.seizure_files]
    files = list(dict.fromkeys(p.resolve() for p in files))
    if len({p.stem for p in files}) != len(files):
        raise ValueError('Duplicate EDF stems would overwrite outputs; run these separately.')
    jobs = [(p, options.resting_state) for p in files]
    if options.interictal_file:
        interictal = Path(options.interictal_file).expanduser().resolve()
        if interictal in files or interictal.stem in {p.stem for p in files}:
            raise ValueError('Interictal EDF must be separate and have a distinct filename stem.')
        jobs.append((interictal, True))
    # Read-only preflight before expensive ICA/connectivity.
    canonical = None
    pair_definitions = None
    for path, is_interictal in jobs:
        if path.suffix.lower() != '.edf' or not path.is_file():
            raise ValueError(f'EDF not found: {path}')
        probe = mne.io.read_raw_edf(path, preload=False)
        probe.drop_channels([ch for ch in options.bad_channels if ch in probe.ch_names])
        names = [ch for ch in probe.ch_names if 'ecg' not in ch.lower() and 'ekg' not in ch.lower()]
        annotation = _read_annotations(path)
        pairs = getBipolarChannels(names, exclude_pairs=annotation.get('bad_channels'))
        definitions = dict(zip(pairs['ch_name'], zip(pairs['anode'], pairs['cathode'])))
        if canonical is None:
            canonical, pair_definitions = list(pairs['ch_name']), definitions
        elif definitions != pair_definitions:
            raise ValueError(f'{path}: bipolar channel sets or endpoint polarity differ across files.')
        if not is_interictal:
            soz = _resolve_soz(annotation['soz'], pairs, canonical)
            if not options.skip_recruitment and not soz:
                raise ValueError(f'{path}: missing SOZ labels; use --skip_recruitment to omit detection.')
        if max(options.ap_range) >= probe.info['sfreq'] / 2:
            raise ValueError(f'{path}: aperiodic fit exceeds Nyquist.')
        if not options.skip_connectivity and any(FREQ_BANDS[b][1] >= probe.info['sfreq'] / 2 for b in options.bands):
            raise ValueError(f'{path}: select connectivity bands below Nyquist.')
        probe.close()

    signals, seizure_sources = [], []
    for path, is_interictal in jobs:
        raw = mne.io.read_raw_edf(path, preload=True)
        raw.drop_channels([ch for ch in options.bad_channels if ch in raw.ch_names])
        annotation = _read_annotations(path)
        raw_bipolar, _ = _preprocess_to_bipolar(raw, annotation.get('bad_channels'), options.skip_ica)
        _align_channels(raw_bipolar, canonical)
        if is_interictal:
            metadata = resting_state(raw_bipolar, path, output_dir, options)
        else:
            metadata, signal = ictal(raw_bipolar, path, output_dir, options, {
                'ch_name': canonical, 'anode': [pair_definitions[ch][0] for ch in canonical],
                'cathode': [pair_definitions[ch][1] for ch in canonical]})
            if options.prototype:
                signals.append(signal)
                seizure_sources.append(str(path))
        metadata['preprocessing'] = {'highpass_hz': 0.5, 'notch_hz': UTILITY_FREQ,
                                     'skip_ica_requested': options.skip_ica,
                                     'ecg_present': 'ecg' in raw.get_channel_types(),
                                     'bad_channels_requested': options.bad_channels}
        with open(output_dir / f'{path.stem}_metadata.json', 'w') as ff:
            json.dump(metadata, ff, indent=2)
        raw.close()
    if options.prototype:
        time_locked_events(signals, output_dir, source.stem.split('seizure')[0],
                           options.bands, seizure_sources)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file')
    parser.add_argument('--resting_state', action='store_true', help='Interictal-only EDF (legacy flag).')
    parser.add_argument('--interictal_file', help='Separate interictal EDF for this patient/implantation.')
    parser.add_argument('--seizure_files', nargs='*', default=[], help='Additional seizure EDFs for this patient.')
    parser.add_argument('--batch_events', action='store_true', help='Discover seizure EDFs with the same prefix.')
    parser.add_argument('--prototype', action='store_true', help='Additionally compute legacy across-seizure estimates.')
    parser.add_argument('--output_dir')
    parser.add_argument('--patient_id', default='', help='Patient identifier added to outputs.')
    parser.add_argument('--bad_channels', nargs='*', default=[], help='Monopolar channels to drop before referencing.')
    parser.add_argument('--preictal_s', type=float, default=30.0)
    parser.add_argument('--window_s', type=float, default=10.0)
    parser.add_argument('--n_cycles', type=float, default=4.0)
    parser.add_argument('--bands', nargs='+', choices=list(FREQ_BANDS), default=list(FREQ_BANDS))
    parser.add_argument('--ap_range', type=float, nargs=2, default=[5.0, 55.0])
    parser.add_argument('--recruitment_band', default='beta', choices=list(FREQ_BANDS))
    parser.add_argument('--recruitment_duration', type=float, default=30.0,
                        help='Seconds after onset; 0 uses seizure offset or EDF end.')
    parser.add_argument('--skip_recruitment', action='store_true')
    parser.add_argument('--skip_connectivity', action='store_true', help='Compute node features/recruitment only.')
    parser.add_argument('--skip_ica', action='store_true')
    options = parser.parse_args()
    if options.window_s <= 0 or options.window_s > 10 or options.preictal_s <= 0 or options.n_cycles <= 0:
        parser.error('Require 0 < window_s <= 10, preictal_s > 0, and n_cycles > 0.')
    if not 0 < options.ap_range[0] < options.ap_range[1] or options.recruitment_duration < 0:
        parser.error('Invalid aperiodic range or recruitment duration.')
    if options.ap_range[0] < UTILITY_FREQ < options.ap_range[1]:
        warnings.warn('Aperiodic fit spans the 60-Hz notch; inspect fits carefully. Default is 5–55 Hz.')
    if options.resting_state and (options.interictal_file or options.batch_events or options.seizure_files or options.prototype):
        parser.error('--resting_state is interictal-only; do not combine it with seizure inputs.')
    if options.prototype and options.skip_connectivity:
        parser.error('--prototype conflicts with --skip_connectivity.')
    return options


if __name__ == '__main__':
    run(_parse_args())

'''
python functional_calc.py "/Users/arjit/Documents/_Lab/thalamic_stim/UChicago_SEEG_Data/electrophys/Seizure & Resting-State Data/iEEG001_SZ_1.EDF" \
  --seizure_files "/Users/arjit/Documents/_Lab/thalamic_stim/UChicago_SEEG_Data/electrophys/Seizure & Resting-State Data/iEEG001_SZ_3.EDF" \
   "/Users/arjit/Documents/_Lab/thalamic_stim/UChicago_SEEG_Data/electrophys/Seizure & Resting-State Data/iEEG001_SZ_5.EDF"\
  --interictal_file \
  "/Users/arjit/Documents/_Lab/thalamic_stim/UChicago_SEEG_Data/electrophys/Seizure & Resting-State Data/iEEG001_RSno1.EDF" \
  --patient_id iEEG001 \
  --output_dir "/Users/arjit/Documents/_Lab/thalamic_stim/UChicago_SEEG_Data/derivatives/iEEG001/seeg/"
'''
