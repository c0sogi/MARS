import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER,
    NUM_TEMPORAL_WINDOWS,
    SAMPLING_RATE,
    SEED,
)

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


def apply_savitzky_golay(x):
    """
    Applies Savitzky-Golay filter to a signal to generate Stream B.
    """
    return signal.savgol_filter(
        x, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER
    )


def extract_intensity(x):
    """
    View 1: Extracts Raw Intensity features (Min, Max, Mean, Std).
    Operates on Stream A (Raw).
    Cite solution_lesson_node_00031
    """
    return np.array([np.min(x), np.max(x), np.mean(x), np.std(x)])


def extract_kinematics(x_smooth):
    """
    View 2: Extracts Robust Kinematics (Velocity, Acceleration).
    Operates on Stream B (Smoothed).
    Cite solution_lesson_node_00026
    """
    # Velocity (1st Derivative)
    vel = np.gradient(x_smooth)
    vel_feats = np.concatenate(
        [np.quantile(vel, [0.01, 0.99]), [np.mean(vel), np.std(vel)]]
    )

    # Acceleration (2nd Derivative)
    acc = np.gradient(vel)
    acc_feats = np.concatenate(
        [np.quantile(acc, [0.01, 0.99]), [np.mean(acc), np.std(acc)]]
    )

    return np.concatenate([vel_feats, acc_feats])


def extract_spectral(x):
    """
    View 4: Extracts Structural Spectral Features (PSD Band Power, Centroid).
    Operates on Stream A (Raw).
    """
    freqs, psd = signal.welch(x, fs=SAMPLING_RATE, nperseg=1024)

    # Band Powers
    # Low (0-2Hz), Mid (2-10Hz), High (10-20Hz)
    feat_psd_low = np.sum(psd[(freqs >= 0) & (freqs < 2)])
    feat_psd_mid = np.sum(psd[(freqs >= 2) & (freqs < 10)])
    feat_psd_high = np.sum(psd[(freqs >= 10) & (freqs < 20)])

    # Spectral Centroid
    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-9)

    return np.array([feat_psd_low, feat_psd_mid, feat_psd_high, spectral_centroid])


def extract_temporal_windows(x):
    """
    View 5: Extracts Flattened Temporal Windows (RMS).
    Operates on Stream A (Raw).
    Cite solution_lesson_node_00010
    """
    wins = np.array_split(x, NUM_TEMPORAL_WINDOWS)
    feats = []
    for w in wins:
        feats.append(np.sqrt(np.mean(w**2)))  # RMS
    return np.array(feats)


def process_segment(df):
    """
    Orchestrates the Dual-Stream feature extraction for a single data segment.

    Args:
        df: DataFrame (60001, 10) of sensor readings.
    Returns:
        1D numpy array of concatenated features.
    """
    # 1. Imputation (Stream A - Raw)
    df = df.fillna(df.mean()).fillna(0)
    raw_data = df.values
    n_sensors = raw_data.shape[1]

    features = []

    # Collection for Stream B (Smoothed) to be used in spatial analysis
    smooth_data_list = []

    # --- Per-Sensor Feature Extraction ---
    for i in range(n_sensors):
        x_raw = raw_data[:, i]

        # 2. Stream B Creation: Smoothing
        x_smooth = apply_savitzky_golay(x_raw)

        # View 1: Raw Intensity
        features.append(extract_intensity(x_raw))

        # View 2: Robust Kinematics
        features.append(extract_kinematics(x_smooth))

        # View 3: Multi-Resolution Texture - Removed
        # features.append(extract_wavelet(x_raw))

        # View 4: Structural Spectral
        features.append(extract_spectral(x_raw))

        # View 5: Flattened Temporal Windows
        features.append(extract_temporal_windows(x_raw))

    # Flatten per-sensor list
    features = np.concatenate(features)

    # --- View 6: Spatial Fingerprinting - Removed
    # df_smooth = pd.DataFrame(np.array(smooth_data_list).T)
    # spatial_feats = extract_spatial(df_smooth)

    return features.astype(np.float32)


def make_dataset(metadata_df, load_cached_data=True, debug_size=None):
    """
    Loads data, computes features using the Dual-Stream pipeline, and handles caching.

    Args:
        metadata_df: DataFrame containing file paths and targets.
        load_cached_data: Boolean, whether to attempt loading from cache.
        debug_size: Int, number of files to process for debugging.

    Returns:
        X (features), y (targets), segment_ids
    """
    # Handle Debugging
    if debug_size:
        metadata_df = metadata_df.iloc[:debug_size]
        print(f"Debug mode: processing {len(metadata_df)} samples.")

    # Determine Cache Filename
    # Use a unique identifier based on the segment IDs in the dataframe
    is_test = "time_to_eruption" not in metadata_df.columns
    subset_id = (
        f"{metadata_df['segment_id'].iloc[0]}_{metadata_df['segment_id'].iloc[-1]}"
    )
    cache_filename = f"features_{subset_id}.parquet"
    if debug_size:
        cache_filename = f"debug_{debug_size}_{cache_filename}"

    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        cached_df = pd.read_parquet(cache_path)

        # Merge with metadata to ensure correct order and target alignment
        merged = metadata_df.merge(cached_df, on="segment_id", how="left")

        feature_cols = [c for c in cached_df.columns if c.startswith("f_")]
        X = merged[feature_cols].values
        segment_ids = merged["segment_id"].values

        if not is_test:
            y = merged["time_to_eruption"].values
            return X, y, segment_ids
        else:
            return X, None, segment_ids

    # 2. Compute Features from Scratch
    print(f"Computing features for {len(metadata_df)} files...")
    X_list = []
    y_list = []
    seg_id_list = []

    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        seg_id = row["segment_id"]

        try:
            # Load Sensor Data
            df = pd.read_csv(file_path, dtype="float32")

            # Extract Features
            feats = process_segment(df)

            X_list.append(feats)
            seg_id_list.append(seg_id)

            if not is_test:
                y_list.append(row["time_to_eruption"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Fallback: Pad with zeros to maintain alignment if a file is corrupt
            if len(X_list) > 0:
                X_list.append(np.zeros_like(X_list[-1]))
            else:
                # If the very first file fails, we can't infer shape. Skip.
                continue

            seg_id_list.append(seg_id)
            if not is_test:
                y_list.append(row["time_to_eruption"])

    X = np.array(X_list)
    segment_ids = np.array(seg_id_list)

    if not is_test:
        y = np.array(y_list)
    else:
        y = None

    # 3. Save to Cache
    print(f"Saving features to {cache_path}...")
    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    cache_df = pd.DataFrame(X, columns=feature_cols)
    cache_df["segment_id"] = segment_ids

    # Optionally save target for verification, though we rely on metadata for labels
    if not is_test:
        cache_df["time_to_eruption"] = y

    cache_df.to_parquet(cache_path)

    return X, y, segment_ids
