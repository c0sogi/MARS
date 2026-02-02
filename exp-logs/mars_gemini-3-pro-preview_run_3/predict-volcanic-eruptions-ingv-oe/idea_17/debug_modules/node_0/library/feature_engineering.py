import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    SAVGOL_WINDOW,
    SAVGOL_POLY,
    SAMPLE_RATE,
    get_mfcc,
)
from library.utils import save_parquet, load_parquet


def get_spectral_centroid(freqs, psd):
    """
    Computes the spectral centroid from frequencies and PSD.
    """
    sum_psd = np.sum(psd)
    if sum_psd == 0:
        return 0.0
    return np.sum(freqs * psd) / sum_psd


def extract_sensor_features(sensor_name, signal, sr=SAMPLE_RATE):
    """
    Extracts features for a single sensor using Augmented Orthogonal Decomposition.
    """
    features = {}

    # --- View A: Trend (Kinematics) ---
    # Savitzky-Golay Filter
    trend = savgol_filter(signal, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)

    # Derivatives
    vel = np.gradient(trend)
    acc = np.gradient(vel)

    # Kinematic Features
    features[f"{sensor_name}_trend_mean"] = np.mean(trend)
    features[f"{sensor_name}_trend_std"] = np.std(trend)

    # Quantiles (1st, 5th, 95th, 99th)
    q01, q05, q95, q99 = np.quantile(trend, [0.01, 0.05, 0.95, 0.99])
    features[f"{sensor_name}_trend_q01"] = q01
    features[f"{sensor_name}_trend_q05"] = q05
    features[f"{sensor_name}_trend_q95"] = q95
    features[f"{sensor_name}_trend_q99"] = q99

    features[f"{sensor_name}_vel_std"] = np.std(vel)
    features[f"{sensor_name}_acc_std"] = np.std(acc)

    # --- View B: Texture (Timbre) ---
    # Residuals
    residual = signal - trend

    # MFCCs (using helper from config)
    mfccs = get_mfcc(residual, sr=sr)
    for i, val in enumerate(mfccs):
        features[f"{sensor_name}_res_mfcc_{i}"] = val

    # Higher-Order Moments
    features[f"{sensor_name}_res_skew"] = skew(residual)
    features[f"{sensor_name}_res_kurt"] = kurtosis(residual)

    # --- View C: Energy / Spectral (Raw) ---
    # Raw Extremes
    features[f"{sensor_name}_raw_min"] = np.min(signal)
    features[f"{sensor_name}_raw_max"] = np.max(signal)
    features[f"{sensor_name}_raw_ptp"] = np.ptp(signal)

    # Spectral Density (Welch)
    freqs, psd = welch(signal, fs=sr, nperseg=1024)
    features[f"{sensor_name}_psd_low"] = np.sum(psd[freqs < 2])
    features[f"{sensor_name}_psd_mid"] = np.sum(psd[(freqs >= 2) & (freqs < 10)])
    features[f"{sensor_name}_psd_high"] = np.sum(psd[freqs >= 10])
    features[f"{sensor_name}_psd_centroid"] = get_spectral_centroid(freqs, psd)

    # Temporal Windows (Flattened)
    chunks = np.array_split(signal, 10)

    for i, chunk in enumerate(chunks):
        c_mean = np.mean(chunk)
        c_rms = np.sqrt(np.mean(chunk**2))
        features[f"{sensor_name}_w{i}_mean"] = c_mean
        features[f"{sensor_name}_w{i}_rms"] = c_rms

    # Return metrics for spatial consistency
    # Using global Mean and global RMS for the sensor
    global_mean = np.mean(signal)
    global_rms = np.sqrt(np.mean(signal**2))

    return features, global_mean, global_rms


def process_segment_file(file_path):
    """
    Loads a file, extracts features for all sensors, and computes spatial consistency.
    """
    try:
        # Load data, using float32 to save memory
        df = pd.read_csv(file_path, dtype="float32")
    except FileNotFoundError:
        return None

    # Imputation: Fill NaNs with column mean
    df = df.fillna(df.mean())

    all_features = {}
    sensor_means = []
    sensor_rms = []

    sensor_cols = [c for c in df.columns if "sensor" in c]

    for sensor in sensor_cols:
        signal = df[sensor].values
        feats, s_mean, s_rms = extract_sensor_features(sensor, signal)
        all_features.update(feats)
        sensor_means.append(s_mean)
        sensor_rms.append(s_rms)

    # --- Spatial Consistency Augmentation ---
    all_features["spatial_mean_std"] = np.std(sensor_means)
    all_features["spatial_mean_range"] = np.ptp(sensor_means)
    all_features["spatial_rms_std"] = np.std(sensor_rms)
    all_features["spatial_rms_range"] = np.ptp(sensor_rms)

    return all_features


def process_dataset(meta_path, cache_filename, load_cached_data=True, debug_size=None):
    """
    Main processing function with caching.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return load_parquet(cache_path)

    print(f"Processing dataset from {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    if debug_size is not None:
        print(f"Debug mode: processing first {debug_size} rows.")
        meta_df = meta_df.head(debug_size)

    features_list = []
    segment_ids = []
    targets = []

    for idx, row in meta_df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        feats = process_segment_file(full_path)
        if feats is not None:
            features_list.append(feats)
            segment_ids.append(row["segment_id"])
            if "time_to_eruption" in row:
                targets.append(row["time_to_eruption"])

    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    df_features["segment_id"] = segment_ids
    if targets:
        df_features["time_to_eruption"] = targets

    # Save to cache
    print(f"Saving features to {cache_path}...")
    save_parquet(df_features, cache_path)

    return df_features
