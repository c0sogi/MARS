import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis, entropy
from joblib import Parallel, delayed
from library.config import Config


def compute_stats(array, prefix):
    """
    Computes basic statistical moments: Mean, Std, Skew, Kurtosis.
    Handles NaNs and Masked arrays safely.
    """
    if len(array) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurt": 0.0,
        }

    # Calculate moments ignoring NaNs
    mu = np.nanmean(array)
    sigma = np.nanstd(array)

    # Skew and Kurtosis
    s = skew(array, nan_policy="omit")
    k = kurtosis(array, nan_policy="omit")

    # Handle masked constants or NaNs returned by scipy.stats
    if np.ma.is_masked(s) or np.isnan(s):
        s = 0.0
    if np.ma.is_masked(k) or np.isnan(k):
        k = 0.0

    return {
        f"{prefix}_mean": mu,
        f"{prefix}_std": sigma,
        f"{prefix}_skew": s,
        f"{prefix}_kurt": k,
    }


def extract_sensor_features(sensor_data, sensor_name):
    """
    Implements the High-Resolution Hybrid-Transform pipeline for a single sensor.
    """
    features = {}

    # Ensure data is float32
    raw = sensor_data.astype(np.float32)

    # ==========================================
    # View A: Trend (Savitzky-Golay)
    # ==========================================
    # Isolate low-frequency baseline drift
    try:
        trend = savgol_filter(
            raw, window_length=Config.SG_WINDOW, polyorder=Config.SG_POLYORDER
        )
    except ValueError:
        # Fallback for very short signals (unlikely given dataset)
        trend = raw

    # Kinematics: 1st and 2nd derivatives
    velocity = np.gradient(trend)
    acceleration = np.gradient(velocity)

    # Extract moments for kinematic signals
    features.update(compute_stats(trend, f"{sensor_name}_trend"))
    features.update(compute_stats(velocity, f"{sensor_name}_vel"))
    features.update(compute_stats(acceleration, f"{sensor_name}_acc"))

    # ==========================================
    # View B: Texture (Residuals)
    # ==========================================
    # Residuals = Raw - Trend. This captures the high-frequency "roughness".
    residuals = raw - trend

    # Energy: Sum of squared residuals
    features[f"{sensor_name}_resid_energy"] = np.sum(residuals**2)

    # Entropy: Shannon entropy of the normalized absolute residuals
    abs_resid = np.abs(residuals)
    sum_abs_resid = np.sum(abs_resid)
    if sum_abs_resid > 0:
        p = abs_resid / sum_abs_resid
        features[f"{sensor_name}_resid_entropy"] = entropy(p)
    else:
        features[f"{sensor_name}_resid_entropy"] = 0.0

    # Higher order moments of residuals
    s_res = skew(residuals, nan_policy="omit")
    k_res = kurtosis(residuals, nan_policy="omit")
    features[f"{sensor_name}_resid_skew"] = (
        0.0 if (np.isnan(s_res) or np.ma.is_masked(s_res)) else s_res
    )
    features[f"{sensor_name}_resid_kurt"] = (
        0.0 if (np.isnan(k_res) or np.ma.is_masked(k_res)) else k_res
    )

    # ==========================================
    # View C: Raw Signal Stats
    # ==========================================
    features[f"{sensor_name}_min"] = np.min(raw)
    features[f"{sensor_name}_max"] = np.max(raw)
    features[f"{sensor_name}_p2p"] = np.ptp(raw)

    # ==========================================
    # View C: High-Resolution Spectral (Welch)
    # ==========================================
    # nperseg=1024 ensures ~0.1 Hz resolution for accurate Low band integration
    f, Pxx = welch(raw, fs=Config.SAMPLING_RATE, nperseg=Config.WELCH_NPERSEG)

    # Determine frequency resolution
    df_freq = f[1] - f[0] if len(f) > 1 else 1.0

    # Integrate Power Spectral Density over defined bands
    for band_name, (low_f, high_f) in Config.FREQ_BANDS.items():
        # Boolean mask for the band
        mask = np.logical_and(f >= low_f, f <= high_f)
        # Integrate: sum(Pxx * df)
        band_power = np.sum(Pxx[mask]) * df_freq
        features[f"{sensor_name}_spec_{band_name}"] = band_power

    # ==========================================
    # View C: Temporal Profiling
    # ==========================================
    # Split signal into non-overlapping windows to capture time evolution
    windows = np.array_split(raw, Config.NUM_TEMPORAL_WINDOWS)
    for i, w in enumerate(windows):
        w_mean = np.mean(w)
        w_rms = np.sqrt(np.mean(w**2))
        features[f"{sensor_name}_w{i+1}_mean"] = w_mean
        features[f"{sensor_name}_w{i+1}_rms"] = w_rms

    return features


def process_segment(segment_id, file_path, target=None):
    """
    Loads a single CSV file, imputes data, and extracts features for all sensors.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    # Load Data
    try:
        df = pd.read_csv(full_path, dtype="float32")
    except FileNotFoundError:
        return None

    # Imputation: Fill NaNs with column mean (preserve DC offset)
    # If a column is all NaN, mean() is NaN, so we fill with 0 afterwards.
    df = df.fillna(df.mean())
    df = df.fillna(0)

    # Initialize feature dictionary
    segment_features = {"segment_id": int(segment_id)}
    if target is not None:
        segment_features["time_to_eruption"] = target

    # Process each sensor column
    for sensor in Config.SENSOR_COLS:
        if sensor in df.columns:
            sensor_feats = extract_sensor_features(df[sensor].values, sensor)
            segment_features.update(sensor_feats)
        else:
            # If sensor is missing from file, we could impute features with 0
            # But based on data analysis, sensors are consistent.
            pass

    return segment_features


def process_dataset_metadata(
    meta_path, dataset_name, load_cached_data=True, debug=False
):
    """
    Orchestrates the feature extraction for an entire dataset defined by metadata.
    Handles Caching and Parallel execution.
    """
    # Define cache path
    cache_file = os.path.join(Config.WORKING_DIR, f"{dataset_name}_features.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {dataset_name} features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Processing {dataset_name} features from scratch...")

    # 2. Load Metadata
    meta_df = pd.read_csv(meta_path)

    # Debug Sampling
    if debug:
        print(
            f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows for {dataset_name}."
        )
        meta_df = meta_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Parallel Processing
    # Using joblib to utilize all vCPUs
    results = Parallel(n_jobs=Config.N_CORES, verbose=0)(
        delayed(process_segment)(
            row["segment_id"],
            row["file_path"],
            row["time_to_eruption"] if "time_to_eruption" in row else None,
        )
        for _, row in meta_df.iterrows()
    )

    # Filter out any failed reads (None)
    results = [r for r in results if r is not None]

    # Convert to DataFrame
    feat_df = pd.DataFrame(results)

    # 4. Save Cache
    print(f"Saving {dataset_name} features to {cache_file}...")
    feat_df.to_parquet(cache_file, index=False)

    return feat_df


def generate_features(load_cached_data=True, debug=False):
    """
    Main entry point. Generates features for Train, Val, and Test sets.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Process Train
    train_df = process_dataset_metadata(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data, debug
    )

    # Process Val
    val_df = process_dataset_metadata(
        Config.VAL_METADATA_PATH, "val", load_cached_data, debug
    )

    # Process Test
    test_df = process_dataset_metadata(
        Config.TEST_METADATA_PATH, "test", load_cached_data, debug
    )

    return train_df, val_df, test_df
