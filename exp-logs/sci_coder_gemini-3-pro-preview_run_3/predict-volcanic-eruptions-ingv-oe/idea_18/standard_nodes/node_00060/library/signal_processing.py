import numpy as np
from scipy import signal
from library import config


def fill_missing_values(x):
    """
    Fills NaN values in the signal with the mean of the non-NaN values.
    This preserves the DC offset of the signal segment.

    Args:
        x (np.ndarray): Input signal array.

    Returns:
        np.ndarray: Signal with NaNs filled.
    """
    if x is None:
        return np.array([])

    if np.isnan(x).any():
        nan_mean = np.nanmean(x)
        # If all values are NaN, fill with 0
        if np.isnan(nan_mean):
            nan_mean = 0.0
        return np.nan_to_num(x, nan=nan_mean)
    return x


def apply_savitzky_golay(x, window_length=None, polyorder=None):
    """
    Applies a Savitzky-Golay filter to the signal to extract the trend (View A).

    Args:
        x (np.ndarray): Input signal.
        window_length (int, optional): The length of the filter window.
                                       Defaults to config.SAVGOL_WINDOW.
        polyorder (int, optional): The order of the polynomial used to fit the samples.
                                   Defaults to config.SAVGOL_POLY.

    Returns:
        np.ndarray: The smoothed signal (trend).
    """
    x_clean = fill_missing_values(x)

    wl = window_length if window_length is not None else config.SAVGOL_WINDOW
    po = polyorder if polyorder is not None else config.SAVGOL_POLY

    # Window length must be odd
    if wl % 2 == 0:
        wl += 1

    # Safety check for short signals
    if len(x_clean) < wl:
        wl = len(x_clean) if len(x_clean) % 2 != 0 else len(x_clean) - 1
        if wl < po:
            return x_clean  # Return raw if signal is too short for filter

    return signal.savgol_filter(x_clean, window_length=wl, polyorder=po)


def compute_derivatives(x):
    """
    Computes the first (velocity) and second (acceleration) derivatives of the signal.

    Args:
        x (np.ndarray): Input signal (typically the Trend view).

    Returns:
        tuple: (velocity, acceleration) as np.ndarrays.
    """
    # np.gradient uses central difference in the interior and first difference at boundaries
    velocity = np.gradient(x)
    acceleration = np.gradient(velocity)
    return velocity, acceleration


def compute_welch_psd(x, fs=None, nperseg=1000):
    """
    Computes the Power Spectral Density (PSD) using Welch's method.

    Args:
        x (np.ndarray): Input signal.
        fs (float, optional): Sampling frequency. Defaults to config.SAMPLE_RATE.
        nperseg (int, optional): Length of each segment. Defaults to 1000 to provide
                                 0.1Hz resolution (fs/nperseg = 100/1000 = 0.1).

    Returns:
        tuple: (frequencies, psd_values)
    """
    x_clean = fill_missing_values(x)
    fs_val = fs if fs is not None else config.SAMPLE_RATE

    # Adjust nperseg if signal is short
    nperseg = min(len(x_clean), nperseg)

    freqs, psd = signal.welch(x_clean, fs=fs_val, nperseg=nperseg)
    return freqs, psd


def compute_band_powers(freqs, psd, bands=None):
    """
    Integrates PSD over specific frequency bands to calculate band power.

    Args:
        freqs (np.ndarray): Array of sample frequencies.
        psd (np.ndarray): Power Spectral Density.
        bands (list, optional): List of tuples (low, high). Defaults to config.FREQ_BANDS.

    Returns:
        dict: Dictionary mapping band names to power values.
    """
    bands_list = bands if bands is not None else config.FREQ_BANDS
    powers = {}

    # Frequency resolution
    if len(freqs) > 1:
        df = freqs[1] - freqs[0]
    else:
        df = 1.0

    for low, high in bands_list:
        # Create boolean mask for the band
        idx = np.logical_and(freqs >= low, freqs <= high)

        if np.sum(idx) > 0:
            # Integrate: Sum(PSD * df)
            band_power = np.sum(psd[idx]) * df
        else:
            band_power = 0.0

        powers[f"band_{low}_{high}"] = band_power

    return powers


def get_signal_stats(x):
    """
    Computes basic statistical descriptors for a signal window.

    Args:
        x (np.ndarray): Input signal.

    Returns:
        dict: Basic stats (mean, std, min, max, rms).
    """
    x_clean = fill_missing_values(x)
    if len(x_clean) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "rms": 0}

    mean_val = np.mean(x_clean)
    std_val = np.std(x_clean)
    min_val = np.min(x_clean)
    max_val = np.max(x_clean)
    rms_val = np.sqrt(np.mean(x_clean**2))

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "rms": rms_val,
    }
