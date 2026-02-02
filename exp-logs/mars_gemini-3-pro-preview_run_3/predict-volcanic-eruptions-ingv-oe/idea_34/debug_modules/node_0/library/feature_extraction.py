import os
import numpy as np
import pandas as pd
from library.config import Config
from library.signal_utils import (
    impute_nans,
    apply_savitzky_golay,
    apply_dwt,
    compute_welch_psd,
    integrate_band_power,
    compute_moments,
    compute_quantiles,
    compute_entropy,
    compute_windowed_diffs,
)


def extract_features_for_sensor(
    signal: np.ndarray, sensor_id: str, cfg: Config
) -> dict:
    """
    Extracts features for a single sensor signal using the Hybrid Decomposition strategy.

    Args:
        signal (np.ndarray): The raw sensor signal.
        sensor_id (str): Identifier for the sensor (e.g., "sensor_1").
        cfg (Config): Configuration object containing hyperparameters.

    Returns:
        dict: A dictionary of extracted features.
    """
    features = {}
    prefix = f"{sensor_id}_"

    # 1. Preprocessing: Impute NaNs
    signal = impute_nans(signal)

    # 2. Decomposition
    # View A: Trend (Savitzky-Golay)
    trend = apply_savitzky_golay(
        signal, window_length=cfg.SAVGOL_WINDOW, polyorder=cfg.SAVGOL_POLY
    )

    # View B: Texture (Wavelet Residuals)
    # Residuals = Raw - Trend
    residuals = signal - trend
    texture = apply_dwt(residuals, wavelet_name=cfg.WAVELET_NAME)

    # View C: Raw
    raw = signal

    # 3. Feature Engineering

    # --- From View A (Trend): Shape via Quantiles ---
    # The trend is smooth, so dense quantiles describe shape well.
    trend_quantiles = compute_quantiles(trend, cfg.QUANTILES)
    for q_name, val in trend_quantiles.items():
        features[f"{prefix}trend_{q_name}"] = val

    # --- From View A (Trend): Kinematics (Derivatives) ---
    # Velocity (1st Derivative)
    vel = np.diff(trend)
    vel_moments = compute_moments(vel)
    for m_name, val in vel_moments.items():
        features[f"{prefix}vel_{m_name}"] = val

    # Acceleration (2nd Derivative)
    acc = np.diff(vel)
    acc_moments = compute_moments(acc)
    for m_name, val in acc_moments.items():
        features[f"{prefix}acc_{m_name}"] = val

    # --- From View B (Texture): Wavelet Stats ---
    features[f"{prefix}txt_energy"] = np.sum(texture**2)
    features[f"{prefix}txt_entropy"] = compute_entropy(texture)

    # Texture moments (Skew/Kurtosis are specifically useful for texture)
    txt_moments = compute_moments(texture)
    features[f"{prefix}txt_skew"] = txt_moments["skew"]
    features[f"{prefix}txt_kurt"] = txt_moments["kurt"]

    # --- From View C (Raw): Absolute Intensity ---
    features[f"{prefix}raw_min"] = np.min(raw)
    features[f"{prefix}raw_max"] = np.max(raw)
    features[f"{prefix}raw_ptp"] = np.ptp(raw)

    # --- From View C (Raw): Spectral Structure (Welch) ---
    freqs, psd = compute_welch_psd(raw, nperseg=cfg.WELCH_NPERSEG)
    for band_name, (low, high) in cfg.FREQ_BANDS.items():
        power = integrate_band_power(freqs, psd, low, high)
        features[f"{prefix}spec_{band_name}"] = power

    # --- From View C (Raw): Differential Temporal Profiling ---
    rms_values, diff_values = compute_windowed_diffs(raw, num_windows=cfg.NUM_WINDOWS)

    # Snapshot features (RMS per window)
    for i, rms in enumerate(rms_values):
        features[f"{prefix}win{i}_rms"] = rms

    # Differential Dynamics (Ramping: diff between consecutive windows)
    for i, diff_val in enumerate(diff_values):
        # diff_values[i] corresponds to window[i+1] - window[i]
        features[f"{prefix}diff_win{i+1}_{i}"] = diff_val

    return features


def process_segment(file_path: str, segment_id: int, cfg: Config) -> dict:
    """
    Loads a CSV and extracts features for all 10 sensors.

    Args:
        file_path (str): Path to the sensor data CSV file.
        segment_id (int): ID of the segment.
        cfg (Config): Configuration object.

    Returns:
        dict: Combined features for the segment, or None if error.
    """
    try:
        # Load data, using float32 to handle potential NaNs and optimize memory
        df = pd.read_csv(file_path, dtype="float32")

        all_features = {}
        all_features["segment_id"] = int(segment_id)

        # Iterate through sensors 1 to 10
        for i in range(1, 11):
            col_name = f"sensor_{i}"
            if col_name in df.columns:
                sensor_feats = extract_features_for_sensor(
                    df[col_name].values, col_name, cfg
                )
                all_features.update(sensor_feats)
            else:
                # If a sensor column is missing, we skip it (or could impute)
                pass

        return all_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_dataset(
    metadata_path: str,
    cfg: Config,
    load_cached_data: bool = True,
    dataset_name: str = "train",
):
    """
    Loads dataset from metadata. Uses caching to avoid re-processing.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cfg (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.
        dataset_name (str): Name of the dataset (train/val/test) for cache naming.

    Returns:
        tuple: (X, y, segment_ids)
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or None): Target variable.
            segment_ids (pd.Series): Segment IDs.
    """
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.WORKING_DIR, f"{dataset_name}_features.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {dataset_name} data from {cache_path}...")
        df = pd.read_parquet(cache_path)

        # Separate X and y
        if "time_to_eruption" in df.columns:
            y = df["time_to_eruption"]
            X = df.drop(columns=["segment_id", "time_to_eruption"])
            return X, y, df["segment_id"]
        else:
            X = df.drop(columns=["segment_id"])
            return X, None, df["segment_id"]

    # 2. Process from Scratch
    print(f"Processing {dataset_name} data from scratch...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    feature_list = []

    # Iterate over metadata
    total = len(meta_df)
    for idx, row in meta_df.iterrows():
        full_path = os.path.join(cfg.INPUT_DIR, row["file_path"])
        feats = process_segment(full_path, row["segment_id"], cfg)

        if feats is not None:
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]
            feature_list.append(feats)

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total} files")

    df = pd.DataFrame(feature_list)

    # Save to cache
    print(f"Saving {dataset_name} data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        X = df.drop(columns=["segment_id", "time_to_eruption"])
        return X, y, df["segment_id"]
    else:
        X = df.drop(columns=["segment_id"])
        return X, None, df["segment_id"]
