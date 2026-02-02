import os
import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
from joblib import Parallel, delayed
import warnings

from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    SAVGOL_WINDOW_LENGTH,
    SAVGOL_POLYORDER,
    N_TEMPORAL_WINDOWS,
    WAVELET_NAME,
    SENSOR_COLS,
    SEED,
)
from library.utils import save_artifact, load_artifact

# Suppress warnings for cleaner output during large-scale processing
warnings.filterwarnings("ignore")


def apply_savgol(data):
    """
    Applies Savitzky-Golay filter to extract trend (View A).
    """
    return signal.savgol_filter(
        data, window_length=SAVGOL_WINDOW_LENGTH, polyorder=SAVGOL_POLYORDER
    )


def get_residuals(raw_data, trend_data):
    """
    Computes residuals to extract texture (View B).
    """
    return raw_data - trend_data


def calculate_rms(x):
    """Calculates Root Mean Square."""
    return np.sqrt(np.mean(x**2))


def extract_distribution_stats(data, prefix):
    """
    Extracts basic distribution statistics: Mean, Std, Min, Max, Quantiles.
    """
    features = {}
    features[f"{prefix}_mean"] = np.mean(data)
    features[f"{prefix}_std"] = np.std(data)
    features[f"{prefix}_min"] = np.min(data)
    features[f"{prefix}_max"] = np.max(data)
    features[f"{prefix}_q01"] = np.quantile(data, 0.01)
    features[f"{prefix}_q05"] = np.quantile(data, 0.05)
    features[f"{prefix}_q95"] = np.quantile(data, 0.95)
    features[f"{prefix}_q99"] = np.quantile(data, 0.99)
    features[f"{prefix}_ptp"] = np.ptp(data)
    features[f"{prefix}_skew"] = stats.skew(data)
    features[f"{prefix}_kurt"] = stats.kurtosis(data)
    return features


def extract_shift_invariant_stats(data, prefix, n_windows):
    """
    Splits signal into n_windows, computes local stats (RMS, Mean),
    and aggregates them to form shift-invariant features.
    """
    # Split data into non-overlapping chunks
    # We use array_split to handle uneven divisions gracefully
    chunks = np.array_split(data, n_windows)

    # Compute metrics for each chunk
    chunk_rms = [calculate_rms(c) for c in chunks]
    chunk_means = [np.mean(c) for c in chunks]
    chunk_stds = [np.std(c) for c in chunks]

    features = {}

    # Aggregated stats of RMS (Volatility of energy)
    features[f"{prefix}_win_rms_mean"] = np.mean(chunk_rms)
    features[f"{prefix}_win_rms_std"] = np.std(chunk_rms)
    features[f"{prefix}_win_rms_min"] = np.min(chunk_rms)
    features[f"{prefix}_win_rms_max"] = np.max(chunk_rms)

    # Aggregated stats of Local Means (Drift stability)
    features[f"{prefix}_win_mean_std"] = np.std(
        chunk_means
    )  # How much the baseline shifts

    # Aggregated stats of Local Stds (Variance stability)
    features[f"{prefix}_win_std_mean"] = np.mean(chunk_stds)
    features[f"{prefix}_win_std_std"] = np.std(chunk_stds)

    return features


def extract_spectral_features(data, prefix):
    """
    Computes PSD and extracts spectral features.
    """
    # Compute Periodogram
    f, Pxx = signal.periodogram(data)

    features = {}
    features[f"{prefix}_spec_mean_power"] = np.mean(Pxx)
    features[f"{prefix}_spec_std_power"] = np.std(Pxx)
    features[f"{prefix}_spec_max_power"] = np.max(Pxx)

    # Dominant Frequency
    features[f"{prefix}_spec_dom_freq"] = f[np.argmax(Pxx)]

    # Spectral Centroid
    if np.sum(Pxx) > 0:
        features[f"{prefix}_spec_centroid"] = np.sum(f * Pxx) / np.sum(Pxx)
    else:
        features[f"{prefix}_spec_centroid"] = 0.0

    return features


def extract_wavelet_features(data, prefix):
    """
    Extracts energy of wavelet detail coefficients.
    """
    features = {}
    try:
        # Decompose
        coeffs = pywt.wavedec(data, WAVELET_NAME, level=2)
        # coeffs[0] is approx (cA2), coeffs[1] is detail 2 (cD2), coeffs[2] is detail 1 (cD1)

        # Energy of details
        features[f"{prefix}_wave_cD1_energy"] = np.sum(coeffs[-1] ** 2)
        features[f"{prefix}_wave_cD2_energy"] = np.sum(coeffs[-2] ** 2)
        features[f"{prefix}_wave_cA2_energy"] = np.sum(coeffs[0] ** 2)
    except Exception:
        # Fallback if wavelet fails (e.g. too short data)
        features[f"{prefix}_wave_cD1_energy"] = 0
        features[f"{prefix}_wave_cD2_energy"] = 0
        features[f"{prefix}_wave_cA2_energy"] = 0

    return features


def process_sensor_channel(data, sensor_name):
    """
    Process a single sensor channel (1D array) through the pipeline.
    """
    # 1. Handle NaNs (Imputation with mean)
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))

    sensor_features = {}

    # --- View C: Energy / Raw ---
    # Global Distribution
    sensor_features.update(extract_distribution_stats(data, f"{sensor_name}_raw"))
    # Shift-Invariant Temporal Stats
    sensor_features.update(
        extract_shift_invariant_stats(data, f"{sensor_name}_raw", N_TEMPORAL_WINDOWS)
    )
    # Spectral Stats
    sensor_features.update(extract_spectral_features(data, f"{sensor_name}"))

    # --- View A: Trend ---
    trend = apply_savgol(data)
    # Basic Trend Stats
    sensor_features.update(extract_distribution_stats(trend, f"{sensor_name}_trend"))

    # Kinematics (Derivatives of Trend)
    velocity = np.gradient(trend)
    acceleration = np.gradient(velocity)

    sensor_features[f"{sensor_name}_vel_mean"] = np.mean(velocity)
    sensor_features[f"{sensor_name}_vel_std"] = np.std(velocity)
    sensor_features[f"{sensor_name}_acc_mean"] = np.mean(acceleration)
    sensor_features[f"{sensor_name}_acc_std"] = np.std(acceleration)

    # --- View B: Texture ---
    texture = get_residuals(data, trend)
    # Higher order moments are crucial for texture
    sensor_features[f"{sensor_name}_txt_skew"] = stats.skew(texture)
    sensor_features[f"{sensor_name}_txt_kurt"] = stats.kurtosis(texture)
    sensor_features[f"{sensor_name}_txt_std"] = np.std(texture)

    # Wavelet Energy on Texture
    sensor_features.update(extract_wavelet_features(texture, f"{sensor_name}_txt"))

    return sensor_features


def process_segment_file(file_path, segment_id):
    """
    Loads a CSV file and generates features for all sensors.
    Returns a dictionary (row) of features.
    """
    try:
        full_path = os.path.join(INPUT_DIR, file_path)
        # Load as float32 to handle NaNs and optimize memory
        df = pd.read_csv(full_path, dtype="float32")

        segment_features = {"segment_id": int(segment_id)}

        # Process each sensor
        for sensor in SENSOR_COLS:
            if sensor in df.columns:
                sensor_data = df[sensor].values
                feats = process_sensor_channel(sensor_data, sensor)
                segment_features.update(feats)
            else:
                # Handle missing sensor column if necessary (unlikely based on data desc)
                pass

        return segment_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def generate_features(metadata_path, dataset_name, load_cached_data=True, debug_n=None):
    """
    Main entry point for feature engineering.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        dataset_name (str): Name tag for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug_n (int, optional): If set, only process the first N files.

    Returns:
        pd.DataFrame: The feature matrix including segment_id and (if available) time_to_eruption.
    """
    # Construct cache path
    cache_filename = f"{dataset_name}_features"
    if debug_n is not None:
        cache_filename += f"_debug_{debug_n}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_filename}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return load_artifact(cache_path)

    print(f"Generating features for {dataset_name} (Debug={debug_n})...")

    # 2. Load Metadata
    meta_df = pd.read_csv(metadata_path)
    if debug_n is not None:
        meta_df = meta_df.iloc[:debug_n]

    # 3. Parallel Processing
    # Use joblib to process files in parallel
    # n_jobs=-1 uses all available cores (12 vCPUs)
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_segment_file)(row["file_path"], row["segment_id"])
        for _, row in meta_df.iterrows()
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # Create DataFrame
    feature_df = pd.DataFrame(results)

    # 4. Merge Target if available
    if "time_to_eruption" in meta_df.columns:
        # Merge on segment_id to attach target
        feature_df = feature_df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 5. Save Cache
    print(f"Saving features to {cache_path}...")
    save_artifact(feature_df, cache_path)

    return feature_df
