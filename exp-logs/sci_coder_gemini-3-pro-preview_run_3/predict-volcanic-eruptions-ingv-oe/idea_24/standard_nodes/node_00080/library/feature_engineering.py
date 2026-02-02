import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from joblib import Parallel, delayed
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SENSOR_COLS,
    SAMPLING_RATE,
    SG_WINDOW,
    SG_POLYORDER,
    HIERARCHICAL_WINDOWS,
    PSD_BANDS,
    N_JOBS,
    SEED,
)

# Ensure reproducibility
np.random.seed(SEED)


def compute_orthogonal_views(x):
    """
    Decomposes the signal into Trend (Low Freq), Texture (High Freq), and Raw (Energy).

    Args:
        x (np.array): Input signal.

    Returns:
        tuple: (trend, texture, raw)
    """
    # Handle NaNs if any remain (though imputation happens at dataframe level)
    if np.isnan(x).any():
        mean_val = np.nanmean(x)
        if np.isnan(mean_val):
            mean_val = 0.0
        x = np.nan_to_num(x, nan=mean_val)

    # View A: Trend (Savitzky-Golay)
    # Isolates low-frequency baseline drift
    trend = signal.savgol_filter(x, window_length=SG_WINDOW, polyorder=SG_POLYORDER)

    # View B: Texture (Residuals)
    # Isolates high-frequency tremors/noise
    texture = x - trend

    # View C: Energy (Raw)
    # Retains absolute intensity
    raw = x

    return trend, texture, raw


def extract_moment_features(x, prefix):
    """
    Extracts higher-order statistical moments.
    """
    if len(x) == 0:
        return {
            f"{prefix}_mean": 0,
            f"{prefix}_std": 0,
            f"{prefix}_skew": 0,
            f"{prefix}_kurt": 0,
        }

    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x),
        f"{prefix}_skew": stats.skew(x),
        f"{prefix}_kurt": stats.kurtosis(x),
    }


def extract_kinematics(trend, prefix):
    """
    Extracts kinematic features (Velocity, Acceleration) from the trend signal.
    """
    # Velocity (1st Derivative)
    vel = np.gradient(trend)
    # Acceleration (2nd Derivative)
    acc = np.gradient(vel)

    feats = {}
    # Moments of Velocity
    feats.update(extract_moment_features(vel, f"{prefix}_vel"))
    # Moments of Acceleration
    feats.update(extract_moment_features(acc, f"{prefix}_acc"))

    return feats


def extract_wavelet_features(texture, prefix):
    """
    Extracts texture features.
    """
    # Decompose
    # Fallback to using texture directly since pywt is not available
    detail_coeffs = texture

    # Energy
    energy = np.sum(detail_coeffs**2) / (len(detail_coeffs) + 1e-9)

    # Entropy (Shannon)
    # Normalize to probability distribution
    p = np.abs(detail_coeffs) / (np.sum(np.abs(detail_coeffs)) + 1e-9)
    entropy = -np.sum(p * np.log(p + 1e-9))

    return {
        f"{prefix}_energy": energy,
        f"{prefix}_entropy": entropy,
        f"{prefix}_skew": stats.skew(detail_coeffs),
        f"{prefix}_kurt": stats.kurtosis(detail_coeffs),
    }


def extract_spectral_features(raw, prefix):
    """
    Extracts Band Power using Welch's Method.
    """
    try:
        freqs, psd = signal.welch(raw, fs=SAMPLING_RATE)

        feats = {}
        for band_name, (low, high) in PSD_BANDS.items():
            # Boolean mask for the band
            idx = np.logical_and(freqs >= low, freqs <= high)

            # Integrate PSD over the band (trapezoidal rule)
            if np.sum(idx) > 0:
                band_power = np.trapz(psd[idx], freqs[idx])
            else:
                band_power = 0.0

            feats[f"{prefix}_psd_{band_name}"] = band_power
        return feats
    except Exception:
        # Fallback if signal is too short or other error
        return {f"{prefix}_psd_{b}": 0.0 for b in PSD_BANDS}


def extract_hierarchical_stats(raw, prefix):
    """
    Splits signal into windows, computes stats per window, then aggregates.
    Captures volatility and evolution without temporal overfitting.
    """
    # Split into non-overlapping windows
    windows = np.array_split(raw, HIERARCHICAL_WINDOWS)

    # Compute window-level statistics
    w_means = np.array([np.mean(w) for w in windows])
    w_rms = np.array([np.sqrt(np.mean(w**2)) for w in windows])

    feats = {}

    # 1. Volatility of Baseline (how much does the mean shift?)
    feats[f"{prefix}_h_mean_std"] = np.std(w_means)
    feats[f"{prefix}_h_mean_range"] = np.max(w_means) - np.min(w_means)

    # 2. Evolution of Energy (is it bursty? constant?)
    feats[f"{prefix}_h_rms_mean"] = np.mean(w_rms)
    feats[f"{prefix}_h_rms_std"] = np.std(w_rms)  # Burstiness
    feats[f"{prefix}_h_rms_max"] = np.max(w_rms)  # Peak intensity

    return feats


def extract_basic_raw_stats(raw, prefix):
    """
    Basic global stats for the raw signal.
    """
    return {
        f"{prefix}_min": np.min(raw),
        f"{prefix}_max": np.max(raw),
        f"{prefix}_p2p": np.max(raw) - np.min(raw),
    }


def process_segment(row):
    """
    Processes a single data segment (file).

    Args:
        row (dict): Row from metadata containing 'segment_id' and 'file_path'.

    Returns:
        dict: Extracted features for the segment.
    """
    file_path = os.path.join(INPUT_DIR, row["file_path"])

    try:
        # Load data
        df = pd.read_csv(file_path, dtype="float32")

        # Imputation: Fill missing values with column mean (preserve DC offset)
        # If a column is entirely NaN, fill with 0
        df = df.fillna(df.mean()).fillna(0)

        features = {}
        features["segment_id"] = int(row["segment_id"])
        if "time_to_eruption" in row:
            features["time_to_eruption"] = row["time_to_eruption"]

        # Process each sensor
        for sensor in SENSOR_COLS:
            if sensor not in df.columns:
                # Should not happen based on analysis, but safe fallback
                continue

            x = df[sensor].values

            # 1. Orthogonal Decomposition
            trend, texture, raw = compute_orthogonal_views(x)

            # 2. Feature Extraction

            # View A: Trend Kinematics & Moments
            # Captures baseline movement and impulsiveness
            features.update(extract_kinematics(trend, f"{sensor}_trend"))
            features.update(extract_moment_features(trend, f"{sensor}_trend"))

            # View B: Texture Wavelets
            # Captures high-frequency structure/noise complexity
            features.update(extract_wavelet_features(texture, f"{sensor}_texture"))

            # View C: Raw Energy, Spectral, Hierarchical
            # Captures absolute intensity and temporal evolution
            features.update(extract_basic_raw_stats(raw, f"{sensor}_raw"))
            features.update(extract_spectral_features(raw, f"{sensor}_spec"))
            features.update(extract_hierarchical_stats(raw, f"{sensor}_hier"))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def generate_features(
    metadata_path, dataset_name, load_cached_data=True, debug_size=None
):
    """
    Main entry point to generate features for a dataset.
    Handles caching, parallelism, and debugging.

    Args:
        metadata_path (str): Path to the metadata CSV.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to load from parquet cache if available.
        debug_size (int, optional): Number of rows to process for debugging.

    Returns:
        pd.DataFrame: DataFrame containing features and target (if available).
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    # Load metadata
    meta_df = pd.read_csv(metadata_path)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {dataset_name} features from {cache_file}...")
        cached_df = pd.read_parquet(cache_file)

        # Cite debug_lesson_3: Validate Cached Artifacts by Schema, Not Data Content
        # We process one sample to ensure the cached features match the current code's output.
        try:
            sample_row = meta_df.iloc[0].to_dict()
            sample_features = process_segment(sample_row)

            if sample_features is not None:
                expected_cols = set(sample_features.keys())
                cached_cols = set(cached_df.columns)

                if expected_cols == cached_cols:
                    print("Cache schema validation passed.")
                    return cached_df
                else:
                    print(
                        f"Cache schema mismatch (Expected {len(expected_cols)} cols, Found {len(cached_cols)} cols). Invalidating cache."
                    )
            else:
                print(
                    "Sample processing failed during cache validation. Invalidating cache."
                )
        except Exception as e:
            print(f"Cache validation failed: {e}. Invalidating cache.")

    # 2. Compute from Scratch
    print(f"Generating features for {dataset_name}...")

    # Apply debug sampling if requested
    if debug_size is not None:
        print(f"Debug Mode: Sampling {debug_size} rows.")
        meta_df = meta_df.iloc[:debug_size]

    # Convert to list of dicts for joblib
    rows = meta_df.to_dict("records")

    # Parallel Processing
    # Use n_jobs from config
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(process_segment)(row) for row in rows
    )

    # Filter out failed segments (None)
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        raise RuntimeError("No features were generated. Check input data paths.")

    feature_df = pd.DataFrame(valid_results)

    # 3. Save Cache
    # Only save if not in debug mode (to avoid overwriting full cache with partial data)
    if debug_size is None:
        print(f"Saving features to {cache_file}...")
        feature_df.to_parquet(cache_file, index=False)

    print(f"Feature generation complete. Shape: {feature_df.shape}")
    return feature_df
