# ecg_utils.py
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import neurokit2 as nk

# --- Filtering ---
def bandpass_filter(x, fs=360, low=0.5, high=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

# --- Normalization ---
def zscore(x):
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# --- Fixed-window segmentation (for training) ---
def make_windows(signal, window_size, step=None):
    if step is None:
        step = window_size
    out = []
    for start in range(0, len(signal) - window_size + 1, step):
        out.append(signal[start:start+window_size])
    return np.asarray(out)

# --- Simple R-peak detection using SciPy ---
def detect_rpeaks_simple(signal, fs=360):
    # Find R-peaks via prominence/height heuristics (works decently for clean signals)
    peaks, _ = find_peaks(signal, distance=int(0.25*fs), prominence=np.std(signal)*0.5)
    return peaks

# --- Robust HR / HRV via NeuroKit2 (fallbacks if fails) ---
def compute_hr_hrv(signal, fs=360):
    try:
        # nk.ecg_process does cleaning + R-peaks + features
        ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
        hr = float(np.nanmedian(ecg_signals["ECG_Rate"])) if "ECG_Rate" in ecg_signals else np.nan
        # Time-domain HRV (RMSSD as example)
        if "RR_Intervals" in info and len(info["RR_Intervals"]) > 2:
            rmssd = float(nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=fs)["HRV_RMSSD"].values[0])
        else:
            rmssd = np.nan
        return hr, rmssd, info.get("ECG_R_Peaks", [])
    except Exception:
        # Fallback: simple SciPy peaks -> HR
        peaks = detect_rpeaks_simple(signal, fs)
        if len(peaks) >= 2:
            rr = np.diff(peaks) / fs  # seconds
            hr = 60.0 / np.median(rr)
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) * 1000.0  # ms
        else:
            hr, rmssd = np.nan, np.nan
        return hr, rmssd, peaks

# --- Basic intervals (very coarse demo; true clinical extraction needs delineation) ---
def estimate_intervals(signal, fs=360, peaks=None):
    # Returns rough QRS width estimate via local slope around peaks (demo only)
    if peaks is None or len(peaks) == 0:
        return np.nan
    widths = []
    halfwin = int(0.08 * fs)  # ~80 ms window
    for p in peaks:
        l = max(p - halfwin, 0)
        r = min(p + halfwin, len(signal))
        seg = signal[l:r]
        # crude width: distance between left and right 50%-of-peak amplitude
        if len(seg) > 2:
            amp = signal[p]
            th = amp * 0.5
            left = l + np.argmax(seg >= th) if np.any(seg >= th) else p
            right = r - (np.argmax(seg[::-1] >= th) + 1) if np.any(seg >= th) else p
            width = (right - left) / fs  # seconds
            widths.append(max(width, 0))
    return float(np.median(widths)) if widths else np.nan
