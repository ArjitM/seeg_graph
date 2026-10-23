"""Seven-feature recruitment detection based on Salami et al. (2025).

Source: https://doi.org/10.1038/s43856-025-00920-9 (Methods and pp. 3-6).
Requires Python 3.10+, NumPy, and SciPy; MNE is optional for the Raw wrapper.

Published decision rules reproduced here:
    * One feature value per second from -30 to +30 seconds around seizure onset.
    * Baseline: -30 to -15 seconds; normalize each channel/feature separately.
    * A feature detects recruitment after three consecutive bins with z > 2.
    * Fewer than two detecting features means no detected spread within 30 s.
    * Otherwise take the earliest qualifying feature's recruitment window.
      The detecting features DO NOT need to cross threshold simultaneously.

Explicit implementation choices where the paper is underspecified:
    * Nonoverlapping 1-second windows, aligned to the seizure-onset sample.
    * Rectified area (integral of |voltage|), sample SD (ddof=1).
    * Hann-window periodogram; absolute, NOT relative or log, band power.
    * Bands: theta [4,8), alpha [8,13), beta [13,30), gamma [30,80) Hz.
    * Frequency-bin summation times frequency spacing integrates each band.
    * All seven features need usable baseline SDs for a final channel label.
      Channels with degenerate baselines are flagged rather than called negative.

Timing convention:
    latency_s is the LEFT edge of the first qualifying 1-second window.
    Early/late labels use latency: < 6 early; >= 6 AND <= 30 late.
    decision_time_s is when enough qualifying 3-second runs have finished.
    For example, feature onsets at 4 and 12 s imply latency_s=4 and
    decision_time_s=15. Recruitment timing is retrospective, with 1-s resolution.

Inputs must already be artifact-screened bipolar signals. This module does not
choose contacts, generate a bipolar montage, filter, notch, or perform ICA.
Apply consistent preprocessing to the continuous data BEFORE extracting epochs;
filtering each 1-second bin separately can introduce edge artifacts.

Detection is per channel. It does not silently pool votes across contacts or
nuclei. Applying the detector to cortical contacts is an extension of the study.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import periodogram


# Keep four bands so there are always exactly seven features. These boundaries
# are our declared choices, not boundaries verified from the authors' code.
DEFAULT_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
FEATURE_NAMES = ("line_length", "area", "voltage_sd", *DEFAULT_BANDS)


def _samples_per_second(sfreq):
    """Require an integer sampling rate to avoid drifting 1-second boundaries."""
    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq < 2:
        raise ValueError("sfreq must be a finite sampling rate >= 2 Hz.")
    samples = int(round(sfreq))
    if not np.isclose(samples, sfreq, rtol=0, atol=1e-9):
        raise ValueError("Use an integer sampling rate, or resample first.")
    return samples


def _first_sustained_window(above, consecutive):
    """Return the first run's starting index, or None if no run qualifies."""
    # Convolution counts True values in each consecutive-bin block. This
    # deliberately requires EVERY bin in the block to exceed the threshold.
    counts = np.convolve(
        np.asarray(above, dtype=np.int64),
        np.ones(consecutive, dtype=np.int64),
        mode="valid",
    )
    hits = np.flatnonzero(counts == consecutive)
    return int(hits[0]) if hits.size else None


def _summarize_zscores(zscores, valid_features, names, threshold, consecutive,
                       min_measures):
    """Apply persistence and consolidation independently to every channel."""
    n_channels = len(names)
    feature_latencies = np.full((n_channels, 7), np.nan)
    rows = []

    for channel, name in enumerate(names):
        for measure in range(7):
            if not valid_features[channel, measure]:
                continue
            # Windows 30:60 are [0,1), [1,2), ..., [29,30) seconds.
            above = zscores[channel, 30:, measure] > threshold
            first = _first_sustained_window(above, consecutive)
            if first is not None:
                feature_latencies[channel, measure] = float(first)

        times = feature_latencies[channel]
        n_triggered = int(np.isfinite(times).sum())
        latency = decision_time = np.nan

        # A flat/broken baseline is a QC failure, not evidence of no recruitment.
        if not valid_features[channel].all():
            status, spread_class = "invalid_baseline", "unclassified"
        elif n_triggered < min_measures:
            status, spread_class = "not_detected", "no_spread_within_30s"
        else:
            status = "detected"
            ordered = np.sort(times[np.isfinite(times)])
            latency = float(ordered[0])
            paper_second = int(latency) + 1
            spread_class = "early" if paper_second <= 6 else "late"
            # Enough evidence exists when the min_measures-th run finishes.
            # This differs from the retrospective onset of the earliest run.
            decision_time = float(ordered[min_measures - 1] + consecutive)

        row = {
            "channel": name,
            "status": status,
            "spread_class": spread_class,
            "latency_s": latency,
            "decision_time_s": decision_time,
            "n_triggered": n_triggered,
        }
        # Keep individual detectors even when the final call is negative or QC
        # fails. They make it possible to audit what produced a recruitment call.
        for measure, feature in enumerate(FEATURE_NAMES):
            row[f"{feature}_latency_s"] = float(times[measure])
            row[f"{feature}_baseline_valid"] = bool(valid_features[channel, measure])
        rows.append(row)

    return rows, feature_latencies


def _detect_recruitment(data, sfreq, onset_s, ch_names=None, *, bands=None,
                       threshold=2.0, consecutive=3, min_measures=2):
    """Extract seven features and detect recruitment for one seizure.

    Parameters
    ----------
    data : array, shape (n_channels, n_samples), or (n_samples,)
        Artifact-screened bipolar signals. MNE voltage units (volts) work as-is.
        This array may be a longer recording or a 60-second epoch.
    sfreq : float
        Sampling rate in Hz; integer-valued rates such as 1000 or 1024 are valid.
    onset_s : float
        Seizure onset in seconds FROM THE FIRST SAMPLE OF THIS ARRAY.
        For a [-30,+30) epoch, pass onset_s=30. For a cropped recording, update
        onset_s to refer to the crop, rather than the original recording.
    ch_names : sequence of str, optional
        Channel names in the same order as data. Defaults to ch_000, ch_001, ...
    bands : dict, optional
        Four (low, high) ranges with keys theta, alpha, beta, gamma.
        Each range includes its lower edge and excludes its upper edge.
    threshold, consecutive, min_measures
        Paper defaults: strict z > 2, three consecutive bins, >=2 detectors.
        Changing them constitutes a modification of the published decision rule.

    Returns
    -------
    dict
        summary: list of per-channel dictionaries, with latencies and labels.
        features, zscores: arrays with shape (channel, 60 windows, 7 features).
        feature_latencies_s, baseline_mean, baseline_sd, valid_features:
            arrays with shape (channel, 7 features).
        bin_start_s, bin_end_s: window coordinates relative to seizure onset.
        settings: parameters and declared implementation choices.

    Notes
    -----
    The full [-30,+30) interval is required, even for shorter seizures. Missing
    values are rejected so they cannot compress time or interrupt the baseline
    unnoticed. Remove bad contacts first and adjudicate artifact-contaminated
    epochs. Signals already elevated at time zero can trigger immediately;
    this algorithm does not require a fresh post-onset threshold crossing.
    """
    samples = _samples_per_second(sfreq)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    if not isinstance(consecutive, (int, np.integer)) or not 1 <= consecutive <= 30:
        raise ValueError("consecutive must be an integer from 1 to 30.")
    if not isinstance(min_measures, (int, np.integer)) or not 1 <= min_measures <= 7:
        raise ValueError("min_measures must be an integer from 1 to 7.")

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must have shape (channels, samples).")
    if not np.isfinite(onset_s):
        raise ValueError("onset_s must be finite.")

    names = ([f"ch_{i:03d}" for i in range(data.shape[0])]
             if ch_names is None else list(ch_names))
    if len(names) != data.shape[0] or not all(isinstance(n, str) for n in names):
        raise ValueError("Supply one string channel name per data row.")
    if len(set(names)) != len(names):
        raise ValueError("Channel names must be unique.")

    # Round onset ONCE to the nearest sample. Build all bin edges from that
    # sample; independently rounding every edge could introduce inconsistencies.
    onset_sample = int(round(float(onset_s) * samples))
    start, stop = onset_sample - 30 * samples, onset_sample + 30 * samples
    if start < 0 or stop > data.shape[1]:
        raise ValueError("Need at least 30 seconds before and after onset.")
    epoch = data[:, start:stop]
    if not np.isfinite(epoch).all():
        raise ValueError("The 60-second epoch contains NaN/Inf or marked artifacts.")
    windows = epoch.reshape(data.shape[0], 60, samples)

    # First three features use the voltage trace, not a spectral estimate.
    features = np.empty((data.shape[0], 60, 7), dtype=float)
    features[..., 0] = np.abs(np.diff(windows, axis=-1)).sum(axis=-1)
    features[..., 1] = np.abs(windows).sum(axis=-1) / samples
    features[..., 2] = windows.std(axis=-1, ddof=1)

    # Compute ALL four powers from a single PSD per channel/window. With a
    # 1-second epoch the natural frequency spacing is 1 Hz (no zero padding).
    frequencies, psd = periodogram(
        windows, fs=float(sfreq), window="hann", detrend="constant",
        scaling="density", axis=-1,
    )
    bands = dict(DEFAULT_BANDS if bands is None else bands)
    if set(bands) != set(DEFAULT_BANDS):
        raise ValueError("bands must contain theta, alpha, beta, gamma exactly.")
    df = frequencies[1] - frequencies[0]
    for feature_index, name in enumerate(DEFAULT_BANDS, start=3):
        low, high = map(float, bands[name])
        if not (np.isfinite(low) and np.isfinite(high)
                and 0 <= low < high <= float(sfreq) / 2):
            raise ValueError(f"Invalid band {name}: {(low, high)} for this sfreq.")
        mask = (frequencies >= low) & (frequencies < high)
        if not mask.any():
            raise ValueError(f"Band {name} contains no sampled frequencies.")
        features[..., feature_index] = psd[..., mask].sum(axis=-1) * df
        bands[name] = (low, high)

    # The first 15 feature windows cover [-30,-15). This is the SD OF THE
    # FEATURE VALUES, not the SD of the underlying voltage samples.
    baseline = features[:, :15, :]
    baseline_mean = baseline.mean(axis=1)
    baseline_sd = baseline.std(axis=1, ddof=1)
    # A relative tolerance makes this check invariant to volts vs microvolts.
    baseline_scale = np.maximum(np.abs(baseline).max(axis=1), np.finfo(float).tiny)
    valid_features = (np.isfinite(baseline_sd)
                      & (baseline_sd > 1e-12 * baseline_scale))
    zscores = np.full_like(features, np.nan)
    np.divide(
        features - baseline_mean[:, None, :], baseline_sd[:, None, :],
        out=zscores, where=valid_features[:, None, :],
    )

    summary, feature_latencies = _summarize_zscores(
        zscores, valid_features, names, float(threshold), int(consecutive),
        int(min_measures),
    )
    return {
        "summary": summary,
        "features": features,
        "zscores": zscores,
        "feature_latencies_s": feature_latencies,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "valid_features": valid_features,
        "ch_names": np.asarray(names),
        "feature_names": np.asarray(FEATURE_NAMES),
        "bin_start_s": np.arange(-30.0, 30.0),
        "bin_end_s": np.arange(-29.0, 31.0),
        "settings": {
            "reference": "https://doi.org/10.1038/s43856-025-00920-9",
            "sfreq": float(sfreq), "onset_s_in_input": float(onset_s),
            "onset_sample": onset_sample,
            "onset_rounding_error_s": onset_sample / samples - float(onset_s),
            "baseline_s": [-30, -15], "search_s": [0, 30],
            "window_s": 1, "threshold_z": float(threshold),
            "consecutive": int(consecutive), "min_measures": int(min_measures),
            "bands_hz": bands, "power_estimator": "Hann periodogram",
            "power_integration": "sum PSD bins * frequency spacing; [low,high)",
            "area": "sum absolute voltage / sfreq", "sd_ddof": 1,
            "latency_convention": "left edge of first qualifying window",
            "early_rule": "latency < 7",
            "baseline_qc": "all seven feature SDs > 1e-12 * baseline scale",
            "numpy_version": np.__version__,
        },
    }


def detect_recruitment(raw_bipolar, onset_s, picks=None, **kwargs):
    """MNE convenience wrapper: read only the required 60 s, not the whole EDF.

    raw_bipolar must already contain bipolar, preprocessed signals. onset_s is
    relative to the FIRST SAMPLE of this Raw object, including after cropping.
    picks is a list of channel names. By default, use all names except info['bads'].
    BAD annotations are retained as NaNs, causing an explicit input/QC error;
    samples are never omitted because omission would corrupt latency estimates.
    """
    sfreq = float(raw_bipolar.info["sfreq"])
    samples = _samples_per_second(sfreq)
    if not np.isfinite(onset_s):
        raise ValueError("onset_s must be finite.")
    onset_sample = int(round(float(onset_s) * samples))
    start, stop = onset_sample - 30 * samples, onset_sample + 30 * samples
    if start < 0 or stop > raw_bipolar.n_times:
        raise ValueError("Raw must include at least 30 s before and after onset.")
    names = ([n for n in raw_bipolar.ch_names if n not in raw_bipolar.info["bads"]]
             if picks is None else list(picks))
    if not names or not all(isinstance(n, str) for n in names):
        raise ValueError("picks must be a nonempty sequence of channel names.")
    marked_bad = set(names).intersection(raw_bipolar.info["bads"])
    if marked_bad:
        raise ValueError(f"Remove marked bad channels from picks: {sorted(marked_bad)}")
    data = raw_bipolar.get_data(
        picks=names, start=start, stop=stop, reject_by_annotation="NaN",
    )
    result = _detect_recruitment(data, sfreq, onset_s=30.0, ch_names=names, **kwargs)
    result["settings"]["raw_onset_s"] = float(onset_s)
    result["settings"]["raw_onset_sample"] = onset_sample
    result["settings"]["raw_onset_rounding_error_s"] = (
        onset_sample / samples - float(onset_s)
    )
    return result


def save_recruitment(result, output_stem):
    """Save a readable summary CSV and numeric details NPZ; return both paths.

    Example: save_recruitment(result, 'results/seizure01') creates
    results/seizure01_summary.csv and results/seizure01_details.npz.
    NPZ stores all features/z-scores, parameters, and timing without raw EEG.
    Load with np.load(path, allow_pickle=False); metadata_json contains settings.
    Files with the same output_stem are overwritten, so use unique seizure IDs.
    """
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_name(stem.name + "_recruitment.csv")
    npz_path = stem.with_name(stem.name + "_recruit_details.npz")
    rows = result["summary"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {key: value for key, value in result.items()
              if isinstance(value, np.ndarray)}
    np.savez_compressed(
        npz_path, **arrays,
        metadata_json=np.asarray(json.dumps(result["settings"], sort_keys=True)),
    )
    return csv_path, npz_path
