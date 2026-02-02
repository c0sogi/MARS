import os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed
from library.config import Config
from library.utils import load_sensor_data


def apply_smoothing(df):
    """
    Applies Savitzky-Golay filter to all sensor columns in the DataFrame.
    """
    # Apply filter to each column
    # axis=0 is the default, which filters along the index (time)
    smoothed_data = scipy.signal.savgol_filter(
        df.values,
        window_length=Config.SAVGOL_WINDOW,
        polyorder=Config.SAVGOL_POLYORDER,
        axis=0,
    )
    return pd.DataFrame(smoothed_data, columns=df.columns, index=df.index)


def extract_kinematics(series, prefix):
    """
    Computes kinematics (velocity, acceleration) and extracts robust statistics.
    """
    features = {}

    # Raw signal (already smoothed)
    s = series.values
    # Velocity (1st derivative)
    v = np.gradient(s)
    # Acceleration (2nd derivative)
    a = np.gradient(v)

    signals = {"raw": s, "vel": v, "acc": a}

    for name, sig in signals.items():
        # Mean and Std
        features[f"{prefix}_{name}_mean"] = np.mean(sig)
        features[f"{prefix}_{name}_std"] = np.std(sig)

        # Quantiles
        quantiles = np.quantile(sig, Config.QUANTILES)
        for q, val in zip(Config.QUANTILES, quantiles):
            q_str = str(int(q * 100)).zfill(2)
            features[f"{prefix}_{name}_q{q_str}"] = val

    return features


def extract_spectral(series, prefix):
    """
    Computes PSD and extracts band power and spectral centroid.
    """
    features = {}
    fs = Config.SAMPLING_RATE

    # Compute PSD using Welch's method
    # Convert to numpy array to prevent Scipy from triggering Pandas MultiIndex lookup (Cite debug_lesson_11)
    s_np = series.values
    freqs, psd = scipy.signal.welch(s_np, fs=fs, nperseg=min(len(s_np), 256))

    # Band Power
    for band_name, (low, high) in Config.FREQ_BANDS.items():
        # Find indices corresponding to the band
        idx = np.logical_and(freqs >= low, freqs <= high)
        # Integrate PSD (approximate with sum)
        band_power = np.sum(psd[idx])
        features[f"{prefix}_spec_band_{band_name}"] = band_power

    # Spectral Centroid
    if np.sum(psd) > 0:
        centroid = np.sum(freqs * psd) / np.sum(psd)
    else:
        centroid = 0.0
    features[f"{prefix}_spec_centroid"] = centroid

    return features


def extract_windowed(series, prefix):
    """
    Splits signal into non-overlapping windows and computes RMS and Mean.
    """
    features = {}

    # Split into N windows
    windows = np.array_split(series, Config.N_WINDOWS)

    for i, w in enumerate(windows):
        # RMS
        rms = np.sqrt(np.mean(w**2))
        features[f"{prefix}_win{i}_rms"] = rms
        features[f"{prefix}_win{i}_mean"] = np.mean(w)
        # Explicitly avoiding Min/Max

    return features


def process_segment(row):
    """
    Processing function for a single row of metadata.
    Loads data, extracts features, returns a dictionary.
    """
    segment_id = row["segment_id"]
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    # 1. Load Data
    df = load_sensor_data(file_path)

    # 2. Imputation (Segment-wise column mean)
    df = df.fillna(df.mean())

    # Handle case where a column might be all NaNs (resulting in NaN mean)
    df = df.fillna(0)

    # 3. Robust Smoothing
    df_smoothed = apply_smoothing(df)

    segment_features = {"segment_id": segment_id}

    # Add target if available
    if "time_to_eruption" in row:
        segment_features["time_to_eruption"] = row["time_to_eruption"]

    # 4. Feature Extraction per Sensor
    for sensor in Config.SENSOR_COLS:
        if sensor not in df_smoothed.columns:
            continue

        s_data = df_smoothed[sensor]
        prefix = sensor

        # View 1: Kinematics
        segment_features.update(extract_kinematics(s_data, prefix))

        # View 2: Spectral
        segment_features.update(extract_spectral(s_data, prefix))

        # View 4: Windowed
        segment_features.update(extract_windowed(s_data, prefix))

    return segment_features


def generate_dataset(meta_path, output_name, load_cached_data=True, debug_size=None):
    """
    Generates the feature dataset from the metadata file.
    Handles caching and parallel processing.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{output_name}.parquet")

    # 1. Load Metadata (moved up for cache validation)
    meta_df = pd.read_csv(meta_path)

    if debug_size is not None:
        meta_df = meta_df.iloc[:debug_size]
        print(f"Debug mode: processing {len(meta_df)} samples.")

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Checking cached features at {cache_path}...")
        cached_df = pd.read_parquet(cache_path)

        # Cite debug_lesson_1: Validate Cached Artifacts Against Source Metadata
        if len(cached_df) == len(meta_df):
            print(
                f"Loading cached features from {cache_path}. Shape: {cached_df.shape}"
            )
            return cached_df
        else:
            print(
                f"Cache mismatch: Found {len(cached_df)} rows, expected {len(meta_df)}. Regenerating..."
            )

    print(f"Generating features for {output_name}...")

    # 3. Parallel Processing
    rows = [row for _, row in meta_df.iterrows()]

    results = Parallel(n_jobs=Config.NUM_WORKERS)(
        delayed(process_segment)(row) for row in rows
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # 4. Create DataFrame
    feature_df = pd.DataFrame(results)

    # 5. Save to Cache
    if not feature_df.empty:
        # Ensure segment_id is int
        feature_df["segment_id"] = feature_df["segment_id"].astype(int)
        feature_df.to_parquet(cache_path, index=False)
        print(f"Features saved to {cache_path}. Shape: {feature_df.shape}")
    else:
        print("Warning: No features generated.")

    return feature_df
