import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from joblib import Parallel, delayed
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    SENSOR_COLS,
    FILL_NA_STRATEGY,
    SG_WINDOW_LENGTH,
    SG_POLYORDER,
    NUM_WINDOWS,
    DEBUG_SAMPLE_SIZE,
    SEED,
)

# Set global seed for reproducibility where applicable
np.random.seed(SEED)


def apply_savitzky_golay(data):
    """
    Applies Savitzky-Golay filter to smooth the data.
    """
    # axis=0 ensures we filter along the time dimension for all sensors
    return signal.savgol_filter(
        data, window_length=SG_WINDOW_LENGTH, polyorder=SG_POLYORDER, axis=0
    )


def compute_kinematics(smoothed_data):
    """
    Computes first (velocity) and second (acceleration) derivatives.
    """
    velocity = np.gradient(smoothed_data, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    return velocity, acceleration


def calculate_stats(array, prefix):
    """
    Calculates basic statistics for a given array.
    Returns a dictionary.
    """
    # Handle potential empty or all-NaN arrays
    if array.size == 0:
        return {}

    # We assume array is 1D here (one sensor's data)
    res = {}
    res[f"{prefix}_mean"] = np.mean(array)
    res[f"{prefix}_std"] = np.std(array)
    res[f"{prefix}_min"] = np.min(array)
    res[f"{prefix}_max"] = np.max(array)
    res[f"{prefix}_skew"] = stats.skew(array, nan_policy="propagate")
    res[f"{prefix}_kurt"] = stats.kurtosis(array, nan_policy="propagate")

    # Quantiles
    q = np.quantile(array, [0.01, 0.05, 0.95, 0.99])
    res[f"{prefix}_q01"] = q[0]
    res[f"{prefix}_q05"] = q[1]
    res[f"{prefix}_q95"] = q[2]
    res[f"{prefix}_q99"] = q[3]

    return res


def compute_spectral_features(x, fs=100.0):
    """
    Computes spectral features using Welch's method.
    """
    f, Pxx = signal.welch(x, fs=fs, nperseg=256)

    # Total Power
    total_power = np.sum(Pxx)

    # Spectral Centroid
    if total_power == 0:
        centroid = 0
    else:
        centroid = np.sum(f * Pxx) / total_power

    # Dominant Frequency
    dominant_freq = f[np.argmax(Pxx)]

    # Band Powers (Simplified relative bins)
    # Low: Bottom 20%, High: Top 20%
    n_bins = len(Pxx)
    low_idx = int(n_bins * 0.2)
    high_idx = int(n_bins * 0.8)

    power_low = np.sum(Pxx[:low_idx])
    power_high = np.sum(Pxx[high_idx:])

    return {
        "spec_centroid": centroid,
        "spec_dom_freq": dominant_freq,
        "spec_power_total": total_power,
        "spec_power_low": power_low,
        "spec_power_high": power_high,
    }


def process_segment(row):
    """
    Processes a single data segment: loads file, cleans, extracts features.
    """
    segment_id = int(row["segment_id"])
    file_path = os.path.join(INPUT_DIR, row["file_path"])

    try:
        # Load Data
        df = pd.read_csv(file_path, dtype="float32")

        # Imputation
        if FILL_NA_STRATEGY == "mean":
            df = df.fillna(df.mean())
        else:
            df = df.fillna(0)

        # Extract raw numpy array for processing
        sensor_data = df[SENSOR_COLS].values

        # 1. Smoothing
        smoothed_data = apply_savitzky_golay(sensor_data)

        # 2. Kinematics
        velocity, acceleration = compute_kinematics(smoothed_data)

        features = {}
        features["segment_id"] = segment_id

        # Iterate over each sensor
        for i, col in enumerate(SENSOR_COLS):
            raw_sig = sensor_data[:, i]
            smooth_sig = smoothed_data[:, i]
            vel_sig = velocity[:, i]
            acc_sig = acceleration[:, i]

            # --- A. Global Stats (Raw) ---
            features.update(calculate_stats(raw_sig, f"{col}_raw"))

            # --- B. Kinematic Stats (Smoothed Derivatives) ---
            features.update(calculate_stats(vel_sig, f"{col}_vel"))
            features.update(calculate_stats(acc_sig, f"{col}_acc"))

            # --- C. Windowed Stats (Raw) ---
            # Split signal into windows
            windows = np.array_split(raw_sig, NUM_WINDOWS)
            for w_idx, window in enumerate(windows):
                features[f"{col}_win{w_idx}_mean"] = np.mean(window)
                features[f"{col}_win{w_idx}_std"] = np.std(window)

            # --- D. Spectral Features (Raw) ---
            spec_feats = compute_spectral_features(raw_sig)
            for k, v in spec_feats.items():
                features[f"{col}_{k}"] = v

        # Add target if available
        if "time_to_eruption" in row:
            features["time_to_eruption"] = row["time_to_eruption"]

        return features

    except Exception as e:
        print(f"Error processing segment {segment_id}: {e}")
        return None


def generate_features(meta_path, output_path, load_cached_data=True):
    """
    Main driver to generate features for a dataset defined by metadata.
    Handles caching and parallel processing.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}...")
        return pd.read_parquet(output_path)

    print(f"Generating features for {meta_path}...")

    # 2. Load Metadata
    meta_df = pd.read_csv(meta_path)

    # Debugging: Limit sample size
    if DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG: Limiting to {DEBUG_SAMPLE_SIZE} samples.")
        meta_df = meta_df.head(DEBUG_SAMPLE_SIZE)

    # 3. Parallel Processing
    # Convert dataframe rows to list of dicts for iteration
    rows = meta_df.to_dict("records")

    # Use all available CPUs
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_segment)(row) for row in rows
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # 4. Aggregate and Save
    feature_df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to Parquet
    feature_df.to_parquet(output_path, index=False)
    print(f"Features saved to {output_path}. Shape: {feature_df.shape}")

    return feature_df


def get_train_val_test_features(load_cached_data=True):
    """
    Public interface to get all feature sets.
    """
    train_df = generate_features(TRAIN_META_PATH, TRAIN_FEATURES_PATH, load_cached_data)
    val_df = generate_features(VAL_META_PATH, VAL_FEATURES_PATH, load_cached_data)
    test_df = generate_features(TEST_META_PATH, TEST_FEATURES_PATH, load_cached_data)

    return train_df, val_df, test_df
