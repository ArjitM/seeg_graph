from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import spectrogram


BandName = Literal["delta", "theta", "alpha", "alpha_theta", "beta", "beta1", "beta2", "gamma", "gamma1", "gamma2", "gamma3"]

BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'alpha_theta': [4, 13],
    'beta': [13, 30],
    'beta1': [13, 18],
    'beta2': [18, 30],
    'gamma': [30, 150],
    'gamma1': [30, 58],
    'gamma2': [62, 118],
    'gamma3': [122, 150],
}



def _band_power_trace(x: np.ndarray, fs: float, band: tuple[float, float], window_s: float = 1.0,
                      step_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding-window log10 band power.
    Returns
    -------
    times_s : ndarray
        Center time of each window.
    log_power : ndarray
        log10 band power for each window.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Expected a finite 1D signal and positive sampling frequency.")
    if not 0 < band[0] < band[1] < fs / 2:
        raise ValueError("Band must satisfy 0 < low < high < Nyquist; bands are not silently truncated.")

    nperseg = int(round(window_s * fs))
    step = int(round(step_s * fs))
    noverlap = nperseg - step

    if len(x) < nperseg:
        raise ValueError("Signal is shorter than window_s.")
    if nperseg < 2:
        raise ValueError("window_s is too short.")
    if step < 1:
        raise ValueError("step_s is too short.")
    if noverlap < 0:
        raise ValueError("step_s cannot exceed window_s.")

    f, t, sxx = spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                            detrend="constant", scaling="density", mode="psd")

    lo, hi = band

    mask = (f >= lo) & (f < hi)

    if mask.sum() < 2:
        raise ValueError(f"Insufficient frequency bins for band {band}. Consider a longer window.")

    if np.__version__.startswith('1'):
        power = np.trapz(sxx[mask], f[mask], axis=0)
    else:
        power = np.trapezoid(sxx[mask], f[mask], axis=0)
    log_power = np.log10(power + np.finfo(float).tiny)

    return t, log_power


def _robust_z(x: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Robust z-score relative to the channel's own baseline."""
    baseline = x[baseline_mask]

    if len(baseline) < 3:
        raise ValueError("Not enough baseline windows.")

    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))  # Median absolute deviation
    scale = 1.4826 * mad

    if scale < np.finfo(float).eps:
        scale = np.std(baseline)

    if scale < np.finfo(float).eps:
        return np.zeros_like(x)

    return (x - med) / scale


def _fill_short_gaps(mask: np.ndarray, max_gap_windows: int) -> np.ndarray:
    """Fill brief False runs enclosed by True samples."""
    mask = mask.astype(bool).copy()

    if max_gap_windows <= 0:
        return mask

    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])

    starts = changes[::2]
    ends = changes[1::2]

    for end_prev, start_next in zip(ends[:-1], starts[1:]):
        if start_next - end_prev <= max_gap_windows:
            mask[end_prev:start_next] = True

    return mask


def _first_sustained_event(z: np.ndarray, times_s: np.ndarray, threshold_z: float, min_duration_s: float,
                           window_s: float, step_s: float, max_gap_s: float, search_start_s: float,
                           search_end_s: float | None = None) -> float | None:
    """Find the first sustained threshold crossing."""
    candidate = z >= threshold_z
    # Search bounds constrain the full spectral window, not only its midpoint.
    candidate &= times_s - window_s / 2 >= search_start_s - 1e-9

    if search_end_s is not None:
        candidate &= times_s + window_s / 2 <= search_end_s + 1e-9

    max_gap_windows = int(np.floor(max_gap_s / step_s))
    candidate = _fill_short_gaps(candidate, max_gap_windows=max_gap_windows)

    if min_duration_s <= window_s:
        min_windows = 1
    else:
        min_windows = int(np.ceil((min_duration_s - window_s) / step_s)) + 1

    padded = np.r_[False, candidate, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])

    starts = changes[::2]
    ends = changes[1::2]

    for start, end in zip(starts, ends):
        if end - start >= min_windows:
            return float(times_s[start])  # Report the window midpoint, without clipping.

    return None


def detect_recruitment(data: np.ndarray, ch_names: Sequence[str], fs: float, soz: Sequence[str],
                       measure: BandName | tuple[float, float] = "beta",
                       baseline_s: tuple[float, float] = (0.0, 30.0), onset_s: float | None = None,
                       window_s: float = 1.0, step_s: float = 0.25, threshold_z: float = 3.0,
                       min_duration_s: float = 2.0, max_gap_s: float = 0.5,
                       search_end_s: float | None = None, return_traces: bool = False):
    """
    Detect spectral recruitment time in every channel.

    Parameters
    ----------
    data
        Shape (n_channels, n_samples).
    ch_names
        Channel names corresponding to the first axis of data.
    fs
        Sampling frequency in Hz.
    soz
        Clinically defined SOZ channel names.
    measure
        One of delta, theta, alpha, alpha_theta, beta, gamma,
        or a custom (low_hz, high_hz) tuple.
    baseline_s
        Baseline interval in seconds from the start of data.
    onset_s
        Known seizure onset in seconds from data start. If None, estimate from earliest sustained
        recruitment among SOZ channels.
    window_s
        Spectral window duration.
    step_s
        Sliding-window step.
    threshold_z
        Robust z-score threshold above baseline.
    min_duration_s
        Minimum duration of sustained recruitment.
    max_gap_s
        Brief below-threshold gaps up to this duration are tolerated.
    search_end_s
        Optional end of recruitment search in seconds from data start.
        Only windows fully contained in the search interval are eligible.
    return_traces
        If True, also return channel-wise z-scored power traces.
    """
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples)")
    if data.shape[0] != len(ch_names):
        raise ValueError("len(ch_names) must equal data.shape[0]")

    if data.shape[0] == 0 or not np.all(np.isfinite(data)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Data must be nonempty and finite; fs must be positive.")
    if len(set(ch_names)) != len(ch_names):
        raise ValueError("Channel names must be unique.")
    if window_s <= 0 or not 0 < step_s <= window_s or min_duration_s <= 0 or max_gap_s < 0:
        raise ValueError("Invalid window, step, duration, or gap parameter.")
    duration = data.shape[1] / fs
    if not 0 <= baseline_s[0] < baseline_s[1] <= duration:
        raise ValueError("Baseline must lie inside the recording.")
    if onset_s is not None and not baseline_s[1] <= onset_s < duration:
        raise ValueError("Annotated onset must follow the baseline and lie inside the recording.")
    if search_end_s is None:
        search_end_s = duration
    start = baseline_s[1] if onset_s is None else onset_s
    if not start < search_end_s <= duration:
        raise ValueError("Search end must follow onset/baseline and lie inside the recording.")
    # Use the actual rounded sample durations for boundary and persistence checks.
    window_s = round(window_s * fs) / fs
    step_s = round(step_s * fs) / fs
    if window_s < 2 / fs or step_s <= 0:
        raise ValueError("Window or step is too short for the sampling rate.")
    onset_reference = "annotated_onset" if onset_s is not None else "detected_SOZ"
    ch_names = list(ch_names)
    soz = set(soz)
    if onset_s is None and not soz:
        raise ValueError("SOZ labels are required to estimate onset.")

    missing_soz = soz.difference(ch_names)
    if missing_soz:
        raise ValueError(f"SOZ channels not present in data: {sorted(missing_soz)}")

    if isinstance(measure, str):
        if measure not in BANDS:
            raise ValueError(f"Unknown measure {measure!r}. Choose from {list(BANDS)} or supply a frequency tuple.")
        band = BANDS[measure]
        measure_name = measure
    else:
        band = tuple(measure)
        measure_name = f"{band[0]:g}-{band[1]:g}_Hz"

    traces = {}
    time_axis = None

    # Compute channel-wise normalized band-power traces
    for ch, x in zip(ch_names, data):
        t, log_power = _band_power_trace(x, fs=fs, band=band, window_s=window_s, step_s=step_s)

        # Exclude baseline windows whose support extends into the seizure.
        baseline_mask = ((t - window_s / 2 >= baseline_s[0] - 1e-9) &
                         (t + window_s / 2 <= baseline_s[1] + 1e-9))
        z = _robust_z(log_power, baseline_mask=baseline_mask)

        traces[ch] = z

        if time_axis is None:
            time_axis = t

    assert time_axis is not None

    # Infer seizure onset from SOZ channels if not provided
    if onset_s is None:
        soz_candidates = []

        for ch in soz:
            candidate = _first_sustained_event(
                traces[ch], times_s=time_axis, threshold_z=threshold_z, min_duration_s=min_duration_s,
                window_s=window_s, step_s=step_s, max_gap_s=max_gap_s,
                search_start_s=baseline_s[1], search_end_s=search_end_s
            )

            if candidate is not None:
                soz_candidates.append(candidate)

        if not soz_candidates:
            raise RuntimeError(
                "No sustained recruitment detected in any SOZ channel. "
                "Check baseline, threshold, measure, or supply onset_s directly."
            )

        onset_s = min(soz_candidates)

    # Detect recruitment in all channels
    rows = []

    for ch in ch_names:
        t_recruit = _first_sustained_event(
            traces[ch], times_s=time_axis, threshold_z=threshold_z, min_duration_s=min_duration_s,
            window_s=window_s, step_s=step_s, max_gap_s=max_gap_s,
            # A detected SOZ reference is already a window midpoint. Include
            # that same window so its latency remains zero.
            search_start_s=(onset_s if onset_reference == "annotated_onset" else
                            max(baseline_s[1], onset_s - window_s / 2)),
            search_end_s=search_end_s
        )

        recruited = t_recruit is not None
        latency = t_recruit - onset_s if recruited else np.nan

        rows.append({
            "Channel": ch,
            "SOZ": ch in soz,
            "Measure": measure_name,
            "Band_low_Hz": band[0],
            "Band_high_Hz": band[1],
            "Recruited": recruited,
            "Recruitment_time_s": t_recruit,
            "Latency_s": latency,
            "Onset_reference": onset_reference,
            "Reference_time_s": onset_s,
            "Latency_from_onset_s": latency if onset_reference == "annotated_onset" else np.nan,
            "Latency_from_SOZ_s": latency if onset_reference == "detected_SOZ" else np.nan,
            "Search_end_s": search_end_s,
            "Status": "recruited" if recruited else "not_detected_in_search_interval",
        })

    results = pd.DataFrame(rows).sort_values(
        ["Recruited", "Latency_s"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    if return_traces:
        trace_df = pd.DataFrame(traces, index=time_axis)
        trace_df.index.name = "Time_s"
        return results, trace_df

    return results