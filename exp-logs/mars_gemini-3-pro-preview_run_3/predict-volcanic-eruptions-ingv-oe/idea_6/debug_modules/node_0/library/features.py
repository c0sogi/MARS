import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SG_WINDOW_LENGTH,
    SG_POLYORDER,
    NUM_TEMPORAL_WINDOWS,
    SENSOR_COLS,
    SAMPLING_RATE,
)


def apply_smoothing(signal):
    """
    Applies Savitzky-Golay filter to a signal array.
    """
    return savgol_filter(signal, window_length=SG_WINDOW_LENGTH, polyorder=SG_POLYORDER)


def compute_basic_stats(array, prefix):
    """
    Computes basic statistics (Mean, Std, Min, Max, Skew, Kurtosis) for an array.
    """
    stats = {}
    stats[f"{prefix}_mean"] = np.mean(array)
    stats[f"{prefix}_std"] = np.std(array)
    stats[f"{prefix}_min"] = np.min(array)
    stats[f"{prefix}_max"] = np.max(array)
    stats[f"{prefix}_skew"] = skew(array)
    stats[f"{prefix}_kurt"] = kurtosis(array)
    # Quantiles
    q = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    stats[f"{prefix}_q05"] = q[0]
    stats[f"{prefix}_q25"] = q[1]
    stats[f"{prefix}_q50"] = q[2]
    stats[f"{prefix}_q75"] = q[3]
    stats[f"{prefix}_q95"] = q[4]
    return stats


def compute_kinematics(df):
    """
    Computes kinematic features (Velocity, Acceleration) and their stats.
    Expects df to be smoothed already or handles smoothing internally.
    Here we assume df contains the smoothed signal columns.
    """
    features = {}

    for col in SENSOR_COLS:
        # 0th Order (Position/Raw Smoothed)
        s_raw = df[col].values
        features.update(compute_basic_stats(s_raw, f"{col}_pos"))

        # 1st Order (Velocity)
        s_vel = np.gradient(s_raw)
        features.update(compute_basic_stats(s_vel, f"{col}_vel"))

        # 2nd Order (Acceleration)
        s_acc = np.gradient(s_vel)
        features.update(compute_basic_stats(s_acc, f"{col}_acc"))

    return features


def compute_windowed_stats(df):
    """
    Divides the signal into non-overlapping windows and computes stats.
    """
    features = {}
    num_rows = len(df)
    window_size = num_rows // NUM_TEMPORAL_WINDOWS

    for col in SENSOR_COLS:
        signal = df[col].values
        for w in range(NUM_TEMPORAL_WINDOWS):
            start = w * window_size
            end = start + window_size if w < NUM_TEMPORAL_WINDOWS - 1 else num_rows
            chunk = signal[start:end]

            # Lightweight stats for windows to avoid feature explosion
            features[f"{col}_w{w}_mean"] = np.mean(chunk)
            features[f"{col}_w{w}_std"] = np.std(chunk)
            features[f"{col}_w{w}_min"] = np.min(chunk)
            features[f"{col}_w{w}_max"] = np.max(chunk)
            # RMS
            features[f"{col}_w{w}_rms"] = np.sqrt(np.mean(chunk**2))

    return features


def compute_spectral(df):
    """
    Computes Spectral features (PSD stats, Centroid, Band Power).
    """
    features = {}

    for col in SENSOR_COLS:
        signal = df[col].values
        # Compute PSD using Welch's method
        freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=256)

        # Normalize PSD to treat as probability distribution for centroid
        psd_norm = psd / (np.sum(psd) + 1e-9)
        spectral_centroid = np.sum(freqs * psd_norm)

        features[f"{col}_spec_centroid"] = spectral_centroid
        features[f"{col}_spec_mean"] = np.mean(psd)
        features[f"{col}_spec_std"] = np.std(psd)
        features[f"{col}_spec_max"] = np.max(psd)

        # Band Power (Simple binning of the PSD)
        # Split PSD into 4 bins
        n_bins = 4
        chunk_size = len(psd) // n_bins
        for b in range(n_bins):
            start = b * chunk_size
            end = start + chunk_size if b < n_bins - 1 else len(psd)
            features[f"{col}_spec_band{b}_energy"] = np.sum(psd[start:end])

    return features


def compute_spatial(df):
    """
    Computes pairwise correlations between sensors.
    """
    features = {}

    # Compute correlation matrix
    corr_matrix = df[SENSOR_COLS].corr()

    # Iterate over upper triangle
    sensors = SENSOR_COLS
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            s1 = sensors[i]
            s2 = sensors[j]
            val = corr_matrix.loc[s1, s2]
            features[f"corr_{s1}_{s2}"] = val

    return features


def extract_features_for_segment(file_path, segment_id):
    """
    Loads a CSV, processes it, and returns a dictionary of features.
    """
    try:
        # Load data (using float32 to save memory)
        df = pd.read_csv(file_path, dtype="float32")

        # 1. Imputation
        # Fill NaNs with column mean to preserve DC offset
        df = df.fillna(df.mean())

        # 2. Smoothing
        # Apply Savitzky-Golay filter to all sensor columns
        # We create a new dataframe for smoothed signals to keep logic clean
        df_smoothed = df.copy()
        for col in SENSOR_COLS:
            df_smoothed[col] = apply_smoothing(df[col].values)

        # 3. Feature Extraction
        features = {}
        features["segment_id"] = segment_id

        # Kinematics (on smoothed data)
        features.update(compute_kinematics(df_smoothed))

        # Windowed Stats (on smoothed data)
        features.update(compute_windowed_stats(df_smoothed))

        # Spectral Features (on smoothed data)
        features.update(compute_spectral(df_smoothed))

        # Spatial Features (on smoothed data)
        features.update(compute_spatial(df_smoothed))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return None or a dict with segment_id and zeros/NaNs?
        # For robustness, we'll re-raise to fail fast in this environment
        raise e


def generate_dataset(metadata_df, dataset_name, load_cached_data=True):
    """
    Generates the feature dataset for a given metadata dataframe.
    Handles caching using Parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing segment_id and file_path.
        dataset_name (str): Name tag for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Feature matrix including segment_id.
    """
    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Compute from Scratch
    print(f"Generating features for {dataset_name} set ({len(metadata_df)} files)...")

    feature_list = []

    # Iterate over metadata
    for _, row in metadata_df.iterrows():
        segment_id = row["segment_id"]
        # Construct full path. Metadata paths are relative to INPUT_DIR
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        features = extract_features_for_segment(full_path, segment_id)
        feature_list.append(features)

    # Create DataFrame
    features_df = pd.DataFrame(feature_list)

    # Ensure segment_id is integer
    if "segment_id" in features_df.columns:
        features_df["segment_id"] = features_df["segment_id"].astype(int)

    # 3. Save Cache
    print(f"Saving features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    features_df.to_parquet(cache_path, index=False)

    return features_df
