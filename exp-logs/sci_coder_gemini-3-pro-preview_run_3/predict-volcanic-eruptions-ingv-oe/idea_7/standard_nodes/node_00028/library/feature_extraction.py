import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed

# Configuration Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_optimized"
N_JOBS = 12
SAVGOL_WINDOW = 25
SAVGOL_POLY = 3
N_WINDOWS = 10
FS = 100  # Assumed sampling frequency for seismic data


def apply_savitzky_golay(data):
    """
    Applies Savitzky-Golay smoothing to the data.
    Args:
        data (np.ndarray): Input data array (n_samples, n_channels).
    Returns:
        np.ndarray: Smoothed data.
    """
    # axis=0 corresponds to the time dimension
    return savgol_filter(
        data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY, axis=0
    )


def compute_kinematics(series, prefix):
    """
    Computes kinematic features (Velocity, Acceleration) and their moments.
    """
    # 1st Derivative (Velocity)
    vel = np.gradient(series)
    # 2nd Derivative (Acceleration)
    acc = np.gradient(vel)

    feats = {}
    # Extract moments for Raw, Velocity, and Acceleration
    for name, data in [("raw", series), ("vel", vel), ("acc", acc)]:
        feats[f"{prefix}_{name}_mean"] = np.mean(data)
        feats[f"{prefix}_{name}_std"] = np.std(data)

        # Robust Quantiles (Cite solution_lesson_node_00026)
        # Replacing unstable higher-order moments (skew/kurt) with explicit percentiles
        q01, q05, q95, q99 = np.quantile(data, [0.01, 0.05, 0.95, 0.99])
        feats[f"{prefix}_{name}_q01"] = q01
        feats[f"{prefix}_{name}_q05"] = q05
        feats[f"{prefix}_{name}_q95"] = q95
        feats[f"{prefix}_{name}_q99"] = q99

        # Robust Range
        feats[f"{prefix}_{name}_range"] = q99 - q01

    return feats


def compute_spectral(series, prefix):
    """
    Computes spectral features using Welch's method.
    """
    # Compute PSD
    f, Pxx = welch(series, fs=FS, nperseg=400)

    # Define bands based on indices (robust to fs assumptions)
    n_bins = len(Pxx)
    idx1 = int(n_bins * 0.2)  # Low freq cutoff
    idx2 = int(n_bins * 0.5)  # Mid freq cutoff

    feats = {}
    feats[f"{prefix}_spec_band_low"] = np.sum(Pxx[:idx1])
    feats[f"{prefix}_spec_band_mid"] = np.sum(Pxx[idx1:idx2])
    feats[f"{prefix}_spec_band_high"] = np.sum(Pxx[idx2:])

    # Spectral Centroid
    total_power = np.sum(Pxx)
    if total_power == 0:
        centroid = 0
    else:
        centroid = np.sum(f * Pxx) / total_power
    feats[f"{prefix}_spec_centroid"] = centroid

    return feats


def compute_temporal_windows(series, prefix):
    """
    Computes statistics over non-overlapping temporal windows.
    """
    # Split signal into N windows
    wins = np.array_split(series, N_WINDOWS)
    feats = {}
    for i, w in enumerate(wins):
        # Robust stats: Mean, Std, RMS (avoid Min/Max)
        w_mean = np.mean(w)
        w_std = np.std(w)
        w_rms = np.sqrt(np.mean(w**2))

        feats[f"{prefix}_win{i}_mean"] = w_mean
        feats[f"{prefix}_win{i}_std"] = w_std
        feats[f"{prefix}_win{i}_rms"] = w_rms

    return feats


def extract_segment_features(segment_id, file_rel_path):
    """
    Worker function to process a single data segment.
    """
    file_path = os.path.join(INPUT_DIR, file_rel_path)

    try:
        # Load data with float32 to handle potential NaNs and memory
        df = pd.read_csv(file_path, dtype="float32")

        # Impute missing values with column mean to preserve DC offsets
        # Handle all-NaN columns by falling back to 0 (Cite debug_lesson_1)
        df = df.fillna(df.mean()).fillna(0)

        # Apply Robust Smoothing
        smoothed_values = apply_savitzky_golay(df.values)
        smoothed_df = pd.DataFrame(smoothed_values, columns=df.columns)

        features = {"segment_id": int(segment_id)}

        # Iterate over all sensors
        for col in df.columns:
            if not col.startswith("sensor"):
                continue

            # Extract signal
            series = smoothed_df[col].values

            # 1. Kinematics (Physics)
            features.update(compute_kinematics(series, col))

            # 2. Spectral (Texture)
            features.update(compute_spectral(series, col))

            # 3. Temporal Evolution (Trend)
            features.update(compute_temporal_windows(series, col))

        return features

    except Exception as e:
        print(f"Error processing segment {segment_id} at {file_path}: {e}")
        return None


def generate_features(
    metadata_path, load_cached_data=True, save_name="features", debug_size=None
):
    """
    Main entry point to generate features for a dataset defined by metadata.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        load_cached_data (bool): If True, attempts to load from cache first.
        save_name (str): Identifier for the cache file (e.g., 'train_features').
        debug_size (int, optional): If set, only process this many files for debugging.

    Returns:
        pd.DataFrame: DataFrame containing features and target (if available).
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{save_name}.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        return pd.read_parquet(cache_file)

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    if debug_size is not None:
        print(f"Debug Mode: Processing first {debug_size} segments.")
        meta_df = meta_df.head(debug_size)

    print(
        f"Starting feature extraction for {len(meta_df)} segments using {N_JOBS} jobs..."
    )

    # 3. Parallel Processing
    results = Parallel(n_jobs=N_JOBS)(
        delayed(extract_segment_features)(row["segment_id"], row["file_path"])
        for _, row in meta_df.iterrows()
    )

    # Filter out any failed segments
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        raise RuntimeError("No features were extracted. Check input data and paths.")

    feat_df = pd.DataFrame(valid_results)

    # 4. Merge Targets (if they exist in metadata)
    if "time_to_eruption" in meta_df.columns:
        # Merge on segment_id to ensure alignment
        feat_df = feat_df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 5. Save to Cache
    print(f"Saving {len(feat_df)} rows to {cache_file}")
    feat_df.to_parquet(cache_file, index=False)

    return feat_df
