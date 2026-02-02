import numpy as np
from scipy import signal
from library.config import Config


def apply_savitzky_golay(x):
    """
    Applies Savitzky-Golay filter to the signal for noise suppression (Stream B).

    Args:
        x (np.ndarray): Input signal array.

    Returns:
        np.ndarray: Smoothed signal.
    """
    return signal.savgol_filter(x, Config.SG_WINDOW, Config.SG_POLY)


def _calculate_stats(x, prefix):
    """
    Helper to calculate standard statistics for a signal array.

    Args:
        x (np.ndarray): Input signal.
        prefix (str): Prefix for the feature keys (e.g., 'sensor_1_smooth').

    Returns:
        dict: Dictionary of statistical features.
    """
    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_q01": np.quantile(x, 0.01),
        f"{prefix}_q05": np.quantile(x, 0.05),
        f"{prefix}_q95": np.quantile(x, 0.95),
        f"{prefix}_q99": np.quantile(x, 0.99),
        f"{prefix}_rms": np.sqrt(np.mean(x**2)),
        f"{prefix}_range": np.max(x) - np.min(x),
    }


def get_raw_extrema(x, prefix):
    """
    Extracts global minimum, maximum, and range from raw data (Stream A).

    Args:
        x (np.ndarray): Raw input signal.
        prefix (str): Sensor prefix (e.g., 'sensor_1').

    Returns:
        dict: Dictionary containing raw extrema features.
    """
    return {
        f"{prefix}_raw_min": np.min(x),
        f"{prefix}_raw_max": np.max(x),
        f"{prefix}_raw_range": np.max(x) - np.min(x),
    }


def get_spectral_features(x, prefix):
    """
    Computes spectral features (Band Power) from raw data (Stream A).

    Args:
        x (np.ndarray): Raw input signal.
        prefix (str): Sensor prefix.

    Returns:
        dict: Dictionary containing spectral power features.
    """
    f, Pxx = signal.periodogram(x, fs=Config.SAMPLING_RATE)
    return {
        f"{prefix}_spec_low": np.sum(Pxx[(f >= 0.1) & (f < 2.0)]),
        f"{prefix}_spec_mid": np.sum(Pxx[(f >= 2.0) & (f < 10.0)]),
        f"{prefix}_spec_high": np.sum(Pxx[(f >= 10.0) & (f < 20.0)]),
    }


def get_temporal_windows(x, prefix):
    """
    Divides signal into non-overlapping windows and computes RMS and Mean (Stream A).

    Args:
        x (np.ndarray): Raw input signal.
        prefix (str): Sensor prefix.

    Returns:
        dict: Dictionary containing flattened temporal window features.
    """
    features = {}
    wins = np.array_split(x, Config.N_TEMPORAL_WINDOWS)
    for w_idx, w_data in enumerate(wins):
        features[f"{prefix}_win{w_idx}_rms"] = np.sqrt(np.mean(w_data**2))
        features[f"{prefix}_win{w_idx}_mean"] = np.mean(w_data)
    return features


def get_kinematics(x_smooth, prefix):
    """
    Computes kinematics (Velocity, Acceleration) and statistics from smoothed data (Stream B).

    Args:
        x_smooth (np.ndarray): Smoothed signal.
        prefix (str): Sensor prefix.

    Returns:
        dict: Dictionary containing kinematic statistics.
    """
    features = {}

    # Derivatives
    vel = np.gradient(x_smooth)
    acc = np.gradient(vel)

    # Statistics on Smoothed Signal and Derivatives
    features.update(_calculate_stats(x_smooth, f"{prefix}_smooth"))
    features.update(_calculate_stats(vel, f"{prefix}_vel"))
    features.update(_calculate_stats(acc, f"{prefix}_acc"))

    return features


def extract_sensor_features(x, sensor_name):
    """
    Wrapper to apply the Dual-Stream pipeline to a single sensor series.

    Args:
        x (np.ndarray): Input signal (usually with NaNs filled).
        sensor_name (str): Name of the sensor (e.g., 'sensor_1').

    Returns:
        dict: Combined dictionary of all features for this sensor.
    """
    features = {}

    # --- Stream A: Raw Data Features ---
    # Captures outliers, true peaks, and spectral content
    features.update(get_raw_extrema(x, sensor_name))
    features.update(get_spectral_features(x, sensor_name))
    features.update(get_temporal_windows(x, sensor_name))

    # --- Stream B: Smoothed Data Features ---
    # Captures robust kinematics (velocity, acceleration) without noise amplification
    x_smooth = apply_savitzky_golay(x)
    features.update(get_kinematics(x_smooth, sensor_name))

    return features
