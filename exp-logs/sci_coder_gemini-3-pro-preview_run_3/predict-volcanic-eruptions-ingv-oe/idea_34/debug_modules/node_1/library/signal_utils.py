import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis
import pywt


def impute_nans(signal: np.ndarray) -> np.ndarray:
    """
    Fills NaN values in the signal with the mean of the non-NaN values.

    Args:
        signal (np.ndarray): Input signal array.

    Returns:
        np.ndarray: Signal with NaNs filled.
    """
    if np.isnan(signal).any():
        return np.nan_to_num(signal, nan=np.nanmean(signal))
    return signal


def apply_savitzky_golay(
    signal: np.ndarray, window_length: int = 51, polyorder: int = 2
) -> np.ndarray:
    """
    Applies a Savitzky-Golay filter to the signal to extract the trend.

    Args:
        signal (np.ndarray): Input signal.
        window_length (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: The filtered signal (trend).
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def apply_dwt(signal: np.ndarray, wavelet_name: str = "db4"):
    """
    Applies Discrete Wavelet Transform and extracts the detail coefficients (texture).

    Args:
        signal (np.ndarray): Input signal (typically residuals).
        wavelet_name (str): Name of the wavelet to use.

    Returns:
        np.ndarray: Detail coefficients (cD) from level 1 decomposition.
    """
    try:
        # pywt.dwt returns (approximation, detail)
        _, cD = pywt.dwt(signal, wavelet_name)
        return cD
    except Exception:
        # Return zeros matching roughly half the length if DWT fails (e.g. signal too short)
        return np.zeros(len(signal) // 2 + 1)


def compute_welch_psd(signal: np.ndarray, fs: float = 100.0, nperseg: int = 1024):
    """
    Computes the Power Spectral Density (PSD) using Welch's method.

    Args:
        signal (np.ndarray): Input signal.
        fs (float): Sampling frequency.
        nperseg (int): Length of each segment.

    Returns:
        tuple: (frequencies, psd_values)
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def integrate_band_power(
    freqs: np.ndarray, psd: np.ndarray, band_low: float, band_high: float
) -> float:
    """
    Calculates the total power within a specific frequency band.

    Args:
        freqs (np.ndarray): Array of sample frequencies.
        psd (np.ndarray): Power Spectral Density values.
        band_low (float): Lower bound of the frequency band.
        band_high (float): Upper bound of the frequency band.

    Returns:
        float: Sum of PSD values within the band.
    """
    idx = np.logical_and(freqs >= band_low, freqs <= band_high)
    return np.sum(psd[idx])


def compute_moments(signal: np.ndarray) -> dict:
    """
    Computes statistical moments: Mean, Std, Skewness, and Kurtosis.

    Args:
        signal (np.ndarray): Input signal.

    Returns:
        dict: Dictionary containing 'mean', 'std', 'skew', 'kurt'.
    """
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "skew": skew(signal),
        "kurt": kurtosis(signal),
    }


def compute_quantiles(signal: np.ndarray, quantiles: list) -> dict:
    """
    Computes specified quantiles of the signal.

    Args:
        signal (np.ndarray): Input signal.
        quantiles (list): List of quantiles to compute (e.g., [0.01, 0.99]).

    Returns:
        dict: Dictionary mapping quantile names (e.g., 'q1', 'q99') to values.
    """
    values = np.quantile(signal, quantiles)
    return {f"q{int(q*100)}": val for q, val in zip(quantiles, values)}


def compute_entropy(signal: np.ndarray) -> float:
    """
    Calculates the Shannon entropy of the signal's energy distribution.

    Args:
        signal (np.ndarray): Input signal.

    Returns:
        float: Entropy value.
    """
    energy = np.sum(signal**2)
    if energy == 0:
        return 0.0
    p = (signal**2) / energy
    # Filter zeros to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def compute_windowed_diffs(signal: np.ndarray, num_windows: int = 10):
    """
    Performs Differential Temporal Profiling.
    Splits signal into windows, computes RMS for each, and calculates
    first-order differences between consecutive window RMS values.

    Args:
        signal (np.ndarray): Input signal.
        num_windows (int): Number of windows to split the signal into.

    Returns:
        tuple: (rms_values, diff_values)
            rms_values (np.ndarray): Array of RMS values for each window.
            diff_values (np.ndarray): Array of differences between consecutive RMS values.
    """
    n = len(signal)
    window_size = max(1, n // num_windows)
    rms_values = []

    for i in range(num_windows):
        start = i * window_size
        # For the last window, extend to the end of the signal
        end = (i + 1) * window_size if i < num_windows - 1 else n

        segment = signal[start:end]
        if len(segment) > 0:
            rms = np.sqrt(np.mean(segment**2))
        else:
            rms = 0.0
        rms_values.append(rms)

    rms_values = np.array(rms_values)
    diff_values = np.diff(rms_values)

    return rms_values, diff_values
