import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import Parallel, delayed
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    SG_WINDOW_SIZE,
    SG_POLY_ORDER,
    WAVELET_TYPE,
    SAMPLING_RATE,
    PSD_BANDS,
    WINDOW_SIZE,
    QUANTILES,
    SEED,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    NUM_SENSORS,
    WORKING_DIR,
)
from library.data_utils import (
    load_metadata,
    load_sensor_segment,
    save_features,
    load_features,
)

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values with the column mean to preserve DC offsets.
    """
    return df.fillna(df.mean())


def extract_trend_features(trend_signal: np.ndarray, sensor_id: str) -> dict:
    """
    Extracts features from the trend signal (View A).
    Includes velocity, acceleration, and distribution statistics.
    """
    features = {}

    # Derivatives
    velocity = np.gradient(trend_signal)
    acceleration = np.gradient(velocity)

    signals = {"trend": trend_signal, "vel": velocity, "acc": acceleration}

    for name, sig in signals.items():
        prefix = f"{sensor_id}_{name}"
        features[f"{prefix}_mean"] = np.mean(sig)
        features[f"{prefix}_std"] = np.std(sig)
        features[f"{prefix}_min"] = np.min(sig)
        features[f"{prefix}_max"] = np.max(sig)

        # Quantiles
        if len(sig) > 0:
            qs = np.quantile(sig, QUANTILES)
            for q, val in zip(QUANTILES, qs):
                features[f"{prefix}_q{int(q*100)}"] = val

    return features


def extract_texture_features(texture_signal: np.ndarray, sensor_id: str) -> dict:
    """
    Extracts features from the texture/residual signal (View B).
    Uses statistical moments of residuals to capture roughness/complexity.
    """
    features = {}
    prefix = f"{sensor_id}_texture"

    # RMS
    rms = np.sqrt(np.mean(texture_signal**2))
    features[f"{prefix}_rms"] = rms

    # Higher order moments
    features[f"{prefix}_std"] = np.std(texture_signal)
    features[f"{prefix}_skew"] = stats.skew(texture_signal)
    features[f"{prefix}_kurt"] = stats.kurtosis(texture_signal)

    # Absolute mean (L1 norm proxy)
    features[f"{prefix}_abs_mean"] = np.mean(np.abs(texture_signal))

    return features


def extract_energy_features(raw_signal: np.ndarray, sensor_id: str) -> dict:
    """
    Extracts energy features from the raw signal (View C).
    """
    features = {}
    prefix = f"{sensor_id}_energy"

    features[f"{prefix}_min"] = np.min(raw_signal)
    features[f"{prefix}_max"] = np.max(raw_signal)
    features[f"{prefix}_ptp"] = np.ptp(raw_signal)  # Peak to peak

    return features


def extract_spectral_features(raw_signal: np.ndarray, sensor_id: str) -> dict:
    """
    Extracts PSD band power features using Welch's method.
    """
    features = {}

    # Compute PSD
    # nperseg set to capture low frequencies (100Hz / 1024 ~= 0.1Hz resolution)
    freqs, psd = signal.welch(raw_signal, fs=SAMPLING_RATE, nperseg=1024)

    # Integrate over bands
    for band_name, (low_f, high_f) in PSD_BANDS.items():
        # Find indices
        idx = np.logical_and(freqs >= low_f, freqs <= high_f)
        # Integrate (simple sum * freq_resolution approximation)
        if np.any(idx):
            band_power = np.trapz(psd[idx], freqs[idx])
        else:
            band_power = 0.0
        features[f"{sensor_id}_psd_{band_name}"] = band_power

    return features


def extract_temporal_stats(raw_signal: np.ndarray, sensor_id: str) -> dict:
    """
    Computes statistics over sliding/non-overlapping windows and aggregates them.
    """
    features = {}
    prefix = f"{sensor_id}_win"

    # Truncate to multiple of WINDOW_SIZE
    n_windows = len(raw_signal) // WINDOW_SIZE
    if n_windows == 0:
        return {}

    reshaped = raw_signal[: n_windows * WINDOW_SIZE].reshape(n_windows, WINDOW_SIZE)

    # Compute stats per window
    win_means = np.mean(reshaped, axis=1)
    win_stds = np.std(reshaped, axis=1)
    win_rms = np.sqrt(np.mean(reshaped**2, axis=1))

    # Aggregate
    features[f"{prefix}_mean_mean"] = np.mean(win_means)
    features[f"{prefix}_mean_std"] = np.std(win_means)
    features[f"{prefix}_std_mean"] = np.mean(win_stds)
    features[f"{prefix}_rms_mean"] = np.mean(win_rms)
    features[f"{prefix}_rms_std"] = np.std(win_rms)

    return features


def process_sensor_column(series: pd.Series, sensor_name: str) -> dict:
    """
    Orchestrates the orthogonal decomposition and feature extraction for a single sensor.
    """
    raw = series.values

    # 1. Decomposition
    # Trend (View A)
    trend = signal.savgol_filter(
        raw, window_length=SG_WINDOW_SIZE, polyorder=SG_POLY_ORDER
    )
    # Texture (View B)
    texture = raw - trend

    # 2. Feature Extraction
    feats = {}
    feats.update(extract_trend_features(trend, sensor_name))
    feats.update(extract_texture_features(texture, sensor_name))
    feats.update(extract_energy_features(raw, sensor_name))
    feats.update(extract_spectral_features(raw, sensor_name))
    feats.update(extract_temporal_stats(raw, sensor_name))

    return feats


def process_row(row: pd.Series) -> dict:
    """
    Processes a single segment (file) defined by a metadata row.
    """
    segment_id = int(row["segment_id"])
    file_path = row["file_path"]

    try:
        # Load and Clean
        df = load_sensor_segment(file_path)
        df = impute_missing(df)

        # Initialize feature dict
        row_features = {"segment_id": segment_id}

        # Add target if available
        if "time_to_eruption" in row:
            row_features["time_to_eruption"] = row["time_to_eruption"]

        # Process each sensor
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

        for col in sensor_cols:
            sensor_feats = process_sensor_column(df[col], col)
            row_features.update(sensor_feats)

        return row_features

    except Exception as e:
        print(f"Error processing segment {segment_id}: {e}")
        return None


def process_dataset(
    split: str, load_cached_data: bool = True, debug: bool = False
) -> pd.DataFrame:
    """
    Main function to process a dataset split (train, val, test).
    Handles caching, parallel processing, and debug sampling.
    """
    # Determine cache path
    if split == "train":
        cache_path = TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = VAL_FEATURES_PATH
    elif split == "test":
        cache_path = TEST_FEATURES_PATH
    else:
        raise ValueError("Unknown split")

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return load_features(cache_path)

    print(f"Processing {split} dataset...")

    # Load Metadata
    meta_df = load_metadata(split)

    # Handle Debugging
    if debug:
        print(f"DEBUG MODE: Sampling {DEBUG_SAMPLE_SIZE} rows.")
        meta_df = meta_df.sample(
            n=min(len(meta_df), DEBUG_SAMPLE_SIZE), random_state=SEED
        )

    # Parallel Processing
    # Convert metadata to list of rows for iteration
    rows = [row for _, row in meta_df.iterrows()]

    # Use joblib for parallel execution
    results = Parallel(n_jobs=-1, verbose=0)(delayed(process_row)(row) for row in rows)

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    if not results:
        raise RuntimeError("No data processed successfully.")

    # Create DataFrame
    feature_df = pd.DataFrame(results)

    # Save to cache (unless in debug mode)
    if not debug:
        print(f"Saving features to {cache_path}...")
        save_features(feature_df, cache_path)

    return feature_df
