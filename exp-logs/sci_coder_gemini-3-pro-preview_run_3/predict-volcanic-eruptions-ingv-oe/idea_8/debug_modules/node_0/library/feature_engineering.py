import os
import numpy as np
import pandas as pd
from scipy import signal

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SEED,
    NUM_SENSORS,
    SENSOR_COLS,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER,
    QUANTILES,
    NUM_WINDOWS,
)
from library.signal_processing import impute_missing_values, apply_savgol_filter


def extract_kinematics(df: pd.DataFrame) -> dict:
    """
    Computes velocity (1st derivative) and acceleration (2nd derivative)
    of the sensor signals, then extracts robust statistical features.

    Features: Mean, Std, Quantiles (defined in config).
    """
    features = {}

    for col in df.columns:
        # Raw signal
        x = df[col].values

        # Velocity (1st derivative)
        v = np.gradient(x)

        # Acceleration (2nd derivative)
        a = np.gradient(v)

        signals = {"raw": x, "vel": v, "acc": a}

        for sig_name, sig_data in signals.items():
            # Mean and Std
            features[f"{col}_{sig_name}_mean"] = np.mean(sig_data)
            features[f"{col}_{sig_name}_std"] = np.std(sig_data)

            # Quantiles
            # QUANTILES is a list like [0.01, 0.05, 0.95, 0.99]
            for q in QUANTILES:
                # Format quantile key, e.g., 0.01 -> q01, 0.95 -> q95
                q_key = f"q{int(q*100):02d}"
                features[f"{col}_{sig_name}_{q_key}"] = np.quantile(sig_data, q)

    return features


def extract_frequency_features(df: pd.DataFrame, fs: int = 100) -> dict:
    """
    Computes Power Spectral Density (PSD) features using Welch's method.
    Extracts Band Power and Spectral Centroid.

    Assumed Sampling Rate (fs): 100 Hz (60000 samples / 600s).
    """
    features = {}

    # Define frequency bands (Hz)
    # Low: Volcanic tremor often < 5Hz
    # Mid: 5-10 Hz
    # High: 10-20 Hz
    # Ultra: > 20 Hz (up to Nyquist 50Hz)
    bands = {
        "low": (0.1, 5.0),
        "mid": (5.0, 10.0),
        "high": (10.0, 20.0),
        "ultra": (20.0, 50.0),
    }

    for col in df.columns:
        x = df[col].values

        # Compute PSD
        # nperseg=256 gives reasonable frequency resolution (~0.4 Hz)
        f, Pxx = signal.welch(x, fs=fs, nperseg=256)

        # Total Power (Energy)
        total_power = np.sum(Pxx)
        # Avoid division by zero
        if total_power < 1e-15:
            total_power = 1e-15

        # Spectral Centroid: Weighted mean of frequencies
        centroid = np.sum(f * Pxx) / total_power
        features[f"{col}_spec_centroid"] = centroid

        # Band Powers
        for band_name, (low_f, high_f) in bands.items():
            # Find indices within band
            idx = np.logical_and(f >= low_f, f <= high_f)
            band_power = np.sum(Pxx[idx])

            features[f"{col}_spec_power_{band_name}"] = band_power
            features[f"{col}_spec_rel_power_{band_name}"] = band_power / total_power

    return features


def extract_temporal_windows(df: pd.DataFrame) -> dict:
    """
    Splits the signal into non-overlapping windows and computes
    flattened statistics (RMS, Mean) for each window to capture temporal evolution.
    """
    features = {}

    # Split dataframe into NUM_WINDOWS chunks
    # np.array_split handles cases where len(df) is not perfectly divisible
    chunks = np.array_split(df, NUM_WINDOWS)

    for w_idx, chunk in enumerate(chunks):
        w_suffix = f"w{w_idx + 1}"

        for col in df.columns:
            x = chunk[col].values

            # Mean
            features[f"{col}_{w_suffix}_mean"] = np.mean(x)

            # RMS (Root Mean Square)
            rms = np.sqrt(np.mean(x**2))
            features[f"{col}_{w_suffix}_rms"] = rms

    return features


def extract_spatial_interactions(df: pd.DataFrame) -> dict:
    """
    Computes pairwise Pearson correlations between all sensors
    to capture spatial synchronization.
    """
    features = {}

    # Compute correlation matrix
    corr_matrix = df.corr(method="pearson")

    cols = df.columns
    n_cols = len(cols)

    # Iterate over the upper triangle of the correlation matrix
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            c1 = cols[i]
            c2 = cols[j]

            # Extract correlation coefficient
            val = corr_matrix.iloc[i, j]

            # Feature name: corr_sensor_1_sensor_2
            features[f"corr_{c1}_{c2}"] = val

    return features


def process_segment(file_path: str) -> dict:
    """
    Orchestrates the feature extraction pipeline for a single data segment.

    1. Load Data
    2. Impute Missing Values
    3. Apply Savitzky-Golay Smoothing
    4. Extract Features (Kinematics, Frequency, Windows, Spatial)
    """
    try:
        # Load CSV
        # Using float32 to optimize memory
        df = pd.read_csv(file_path, dtype="float32")

        # --- Preprocessing ---
        df = impute_missing_values(df)
        df = apply_savgol_filter(df)

        # --- Feature Extraction ---
        # 1. Robust Kinematics (Vel, Acc, Quantiles)
        feat_kin = extract_kinematics(df)

        # 2. Structural Spectral Features (PSD Bands)
        feat_freq = extract_frequency_features(df)

        # 3. Flattened Robust Windows (Temporal Evolution)
        feat_win = extract_temporal_windows(df)

        # 4. Spatial Interactions (Correlations)
        feat_spatial = extract_spatial_interactions(df)

        # Combine all feature dictionaries
        all_features = {**feat_kin, **feat_freq, **feat_win, **feat_spatial}

        return all_features

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def generate_features(
    metadata_path: str,
    output_name: str,
    load_cached_data: bool = True,
    debug_size: int = None,
) -> pd.DataFrame:
    """
    Generates features for all segments listed in the metadata file.
    Implements caching using Parquet.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        output_name (str): Name for the cached output file (e.g., 'train_features').
        load_cached_data (bool): If True, attempts to load from cache first.
        debug_size (int, optional): If set, only processes N rows for debugging.

    Returns:
        pd.DataFrame: DataFrame containing features, segment_id, and time_to_eruption (if available).
    """
    cache_path = os.path.join(WORKING_DIR, f"{output_name}.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Starting feature generation for {output_name}...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    if debug_size is not None:
        print(f"Debug Mode: Processing first {debug_size} segments.")
        meta_df = meta_df.head(debug_size)

    # 3. Iterate and Process
    feature_list = []
    segment_ids = []
    targets = []

    total_files = len(meta_df)

    for idx, row in meta_df.iterrows():
        segment_id = int(row["segment_id"])

        # Construct full file path
        # Metadata 'file_path' is relative to INPUT_DIR (e.g., 'train/123.csv')
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Process
        features = process_segment(full_path)

        if features is not None:
            feature_list.append(features)
            segment_ids.append(segment_id)

            # Handle target variable
            if "time_to_eruption" in row:
                targets.append(row["time_to_eruption"])
            else:
                targets.append(np.nan)

        # Simple progress log
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total_files} segments...")

    # 4. Aggregate into DataFrame
    if not feature_list:
        print("Warning: No features were generated.")
        return pd.DataFrame()

    features_df = pd.DataFrame(feature_list)
    features_df["segment_id"] = segment_ids

    # Include target column (will be NaN for test set)
    features_df["time_to_eruption"] = targets

    # 5. Save to Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    features_df.to_parquet(cache_path, index=False)
    print(f"Features saved to {cache_path}")

    return features_df
