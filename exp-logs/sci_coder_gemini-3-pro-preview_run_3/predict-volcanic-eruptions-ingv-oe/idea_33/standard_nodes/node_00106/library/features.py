import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from joblib import Parallel, delayed
from library.config import PATHS, SIGNAL_PARAMS, FEATURE_PARAMS, COMPUTE_PARAMS


def clean_signal(df):
    """
    Imputes missing values with the column mean to preserve DC offsets.
    """
    return df.fillna(df.mean()).fillna(0)


def decompose_signal(series):
    """
    Performs Pyramidal Orthogonal Decomposition.
    Returns:
        trend: Savitzky-Golay filtered signal (View A)
        residuals: Raw - Trend
        raw: The original signal (View C)
    """
    raw = series.values

    # View A: Trend via Savitzky-Golay
    # Strict physics-based constraints: Large window, Quadratic order (Order 2)
    trend = signal.savgol_filter(
        raw, window_length=SIGNAL_PARAMS.SG_WINDOW, polyorder=SIGNAL_PARAMS.SG_ORDER
    )

    # View B: Residuals for Texture analysis
    residuals = raw - trend

    return trend, residuals, raw


def get_trend_features(trend, sensor_id):
    """
    Extracts dense grid of quantiles from the Trend signal (View A).
    Split-Granularity: Trend is smooth, so quantiles describe shape accurately.
    """
    feats = {}
    quantiles = np.quantile(trend, FEATURE_PARAMS.QUANTILES)
    for q, val in zip(FEATURE_PARAMS.QUANTILES, quantiles):
        feats[f"{sensor_id}_trend_q{int(q*100)}"] = val
    return feats


def get_kinematic_features(trend, sensor_id):
    """
    Extracts kinematic moments and quantiles from Trend derivatives (View A).
    """
    feats = {}

    # First Derivative (Velocity)
    velocity = np.gradient(trend)
    feats[f"{sensor_id}_vel_mean"] = np.mean(velocity)
    feats[f"{sensor_id}_vel_std"] = np.std(velocity)
    feats[f"{sensor_id}_vel_skew"] = np.nan_to_num(stats.skew(velocity))
    feats[f"{sensor_id}_vel_kurt"] = np.nan_to_num(stats.kurtosis(velocity))

    # Add granular quantiles for velocity
    # Cite solution_lesson_node_00059: Granular distributional statistics on derivatives
    vel_quantiles = np.quantile(velocity, FEATURE_PARAMS.QUANTILES)
    for q, val in zip(FEATURE_PARAMS.QUANTILES, vel_quantiles):
        feats[f"{sensor_id}_vel_q{int(q*100)}"] = val

    # Second Derivative (Acceleration)
    acceleration = np.gradient(velocity)
    feats[f"{sensor_id}_acc_mean"] = np.mean(acceleration)
    feats[f"{sensor_id}_acc_std"] = np.std(acceleration)
    feats[f"{sensor_id}_acc_skew"] = np.nan_to_num(stats.skew(acceleration))
    feats[f"{sensor_id}_acc_kurt"] = np.nan_to_num(stats.kurtosis(acceleration))

    # Add granular quantiles for acceleration
    acc_quantiles = np.quantile(acceleration, FEATURE_PARAMS.QUANTILES)
    for q, val in zip(FEATURE_PARAMS.QUANTILES, acc_quantiles):
        feats[f"{sensor_id}_acc_q{int(q*100)}"] = val

    return feats


def get_texture_features(residuals, sensor_id):
    """
    Extracts texture statistics from Residuals (View B).
    Cite solution_lesson_node_00049: Statistical moments specifically on Texture.
    """
    feats = {}
    feats[f"{sensor_id}_resid_rms"] = np.sqrt(np.mean(residuals**2))
    feats[f"{sensor_id}_resid_mean"] = np.mean(residuals)
    feats[f"{sensor_id}_resid_std"] = np.std(residuals)
    feats[f"{sensor_id}_resid_skew"] = np.nan_to_num(stats.skew(residuals))
    feats[f"{sensor_id}_resid_kurt"] = np.nan_to_num(stats.kurtosis(residuals))

    # Robust extremes
    feats[f"{sensor_id}_resid_q01"] = np.quantile(residuals, 0.01)
    feats[f"{sensor_id}_resid_q99"] = np.quantile(residuals, 0.99)

    return feats


def get_absolute_intensity_features(raw, sensor_id):
    """
    Extracts global intensity metrics from Raw signal (View C).
    """
    feats = {}
    feats[f"{sensor_id}_raw_min"] = np.min(raw)
    feats[f"{sensor_id}_raw_max"] = np.max(raw)
    feats[f"{sensor_id}_raw_ptp"] = np.ptp(raw)
    return feats


def get_spectral_features(raw, sensor_id):
    """
    Extracts High-Resolution Spectral Structure (View C).
    Uses Welch's method with high nperseg to resolve low frequencies.
    """
    feats = {}

    # Welch's Method
    freqs, psd = signal.welch(
        raw, fs=SIGNAL_PARAMS.SAMPLING_RATE, nperseg=SIGNAL_PARAMS.WELCH_NPERSEG
    )

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Integrate power in bands
    for band_name, (low_f, high_f) in SIGNAL_PARAMS.FREQ_BANDS.items():
        # Create mask for the band
        mask = (freqs >= low_f) & (freqs <= high_f)
        if np.sum(mask) > 0:
            # Integrate PSD: sum(Pxx) * df
            band_power = np.sum(psd[mask]) * freq_res
        else:
            band_power = 0.0
        feats[f"{sensor_id}_spec_{band_name}"] = band_power

    return feats


def get_temporal_features(raw, sensor_id):
    """
    Extracts Aggregated Temporal Statistics (View C).
    Cite solution_lesson_node_00050: Shift Invariance via Aggregation.
    """
    feats = {}

    # Split into non-overlapping windows
    windows = np.array_split(raw, FEATURE_PARAMS.N_TEMPORAL_SEGMENTS)

    # Calculate window statistics
    rms_values = [np.sqrt(np.mean(w**2)) for w in windows]
    mean_values = [np.mean(w) for w in windows]

    # Aggregate statistics across windows (Shift Invariance)
    for name, vals in [("rms", rms_values), ("mean", mean_values)]:
        feats[f"{sensor_id}_win_{name}_mean"] = np.mean(vals)
        feats[f"{sensor_id}_win_{name}_std"] = np.std(vals)
        feats[f"{sensor_id}_win_{name}_min"] = np.min(vals)
        feats[f"{sensor_id}_win_{name}_max"] = np.max(vals)
        feats[f"{sensor_id}_win_{name}_range"] = np.ptp(vals)

    return feats


def process_segment(file_path):
    """
    Loads a single CSV file and extracts all features for all sensors.
    """
    try:
        # Load data
        # Using float32 to match dataset description note
        df = pd.read_csv(file_path, dtype="float32")

        # Clean data
        df = clean_signal(df)

        segment_features = {}

        # Process each sensor
        for sensor in FEATURE_PARAMS.SENSORS:
            if sensor not in df.columns:
                continue

            # Decompose
            trend, residuals, raw = decompose_signal(df[sensor])

            # Feature Extraction Pipeline
            # 1. Trend Shape (View A)
            segment_features.update(get_trend_features(trend, sensor))

            # 2. Kinematics (View A)
            segment_features.update(get_kinematic_features(trend, sensor))

            # 3. Texture (View B)
            segment_features.update(get_texture_features(residuals, sensor))

            # 4. Absolute Intensity (View C)
            segment_features.update(get_absolute_intensity_features(raw, sensor))

            # 5. Spectral Structure (View C)
            segment_features.update(get_spectral_features(raw, sensor))

            # 6. Temporal Profiling (View C)
            segment_features.update(get_temporal_features(raw, sensor))

        return segment_features

    except Exception as e:
        # In case of read error, return empty or handle gracefully
        # For this task, we assume data integrity but print error
        print(f"Error processing {file_path}: {e}")
        return {}


def generate_features(metadata_path, dataset_name, load_cached_data=True):
    """
    Main entry point to generate features for a dataset (train/val/test).
    Handles caching and parallel processing.
    """
    cache_file = os.path.join(PATHS.WORKING_DIR, f"{dataset_name}_features.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating features for {dataset_name}...")

    # 2. Load Metadata
    meta_df = pd.read_csv(metadata_path)

    # 3. Parallel Processing
    # Construct full paths
    file_paths = [os.path.join(PATHS.INPUT_DIR, fp) for fp in meta_df["file_path"]]

    # Execute
    results = Parallel(n_jobs=COMPUTE_PARAMS.N_JOBS, verbose=0)(
        delayed(process_segment)(fp) for fp in file_paths
    )

    # 4. Construct DataFrame
    features_df = pd.DataFrame(results)

    # Add segment_id and target if available
    features_df["segment_id"] = meta_df["segment_id"]
    if "time_to_eruption" in meta_df.columns:
        features_df["time_to_eruption"] = meta_df["time_to_eruption"]

    # 5. Save to Cache
    print(f"Saving features to {cache_file}...")
    features_df.to_parquet(cache_file, index=False)

    print(f"Feature generation complete. Shape: {features_df.shape}")
    return features_df
