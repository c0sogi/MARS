import os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed
from library import config, utils

# Ensure working directory exists for caching
os.makedirs(config.WORKING_DIR, exist_ok=True)


def extract_kinematics(signal):
    """
    View A (Trend): Computes kinematics (velocity, acceleration) and higher-order moments.
    """
    # First derivative (Velocity)
    velocity = np.gradient(signal)
    # Second derivative (Acceleration)
    acceleration = np.gradient(velocity)

    features = {}

    # Function to compute stats for a specific kinematic component
    def compute_moments(arr, prefix):
        features[f"{prefix}_mean"] = np.mean(arr)
        features[f"{prefix}_std"] = np.std(arr)
        features[f"{prefix}_skew"] = scipy.stats.skew(arr)
        features[f"{prefix}_kurt"] = scipy.stats.kurtosis(arr)
        features[f"{prefix}_max"] = np.max(arr)
        features[f"{prefix}_min"] = np.min(arr)

    compute_moments(signal, "trend")
    compute_moments(velocity, "vel")
    compute_moments(acceleration, "acc")

    return features


def extract_wavelet_stats(signal):
    """
    View B (Texture): Extracts statistics from Discrete Wavelet Transform detail coefficients.
    STUBBED: pywt is not available in this environment.
    """
    return {}


def extract_high_res_psd(signal):
    """
    View C (Spectral): Computes PSD Band Power using Welch's method with high resolution.
    """
    freqs, psd = scipy.signal.welch(
        signal, fs=config.SAMPLING_RATE, nperseg=config.PSD_NPERSEG
    )

    features = {}

    for band_name, (low_f, high_f) in config.PSD_BANDS.items():
        # Find indices corresponding to the band
        idx = np.logical_and(freqs >= low_f, freqs <= high_f)

        # Integrate power in this band (using Simpson's rule or simple sum * df)
        # Here we use simple summation approximation as freq bins are uniform
        band_power = np.sum(psd[idx])
        features[f"psd_{band_name}_power"] = band_power

        if np.sum(idx) > 0:
            features[f"psd_{band_name}_peak"] = np.max(psd[idx])
            features[f"psd_{band_name}_mean"] = np.mean(psd[idx])
        else:
            features[f"psd_{band_name}_peak"] = 0
            features[f"psd_{band_name}_mean"] = 0

    return features


def extract_temporal_windows(signal):
    """
    View C (Temporal): Flattens signal into non-overlapping windows to capture time evolution.
    """
    windows = np.array_split(signal, config.TEMPORAL_WINDOW_COUNT)
    features = {}

    for i, w in enumerate(windows):
        features[f"win_{i}_mean"] = np.mean(w)
        features[f"win_{i}_std"] = np.std(w)
        features[f"win_{i}_rms"] = np.sqrt(np.mean(w**2))
        features[f"win_{i}_min"] = np.min(w)
        features[f"win_{i}_max"] = np.max(w)

    return features


def extract_basic_stats(signal):
    """
    View C (Absolute Intensity): Basic statistics on raw signal.
    """
    features = {}
    features["raw_min"] = np.min(signal)
    features["raw_max"] = np.max(signal)
    features["raw_ptp"] = np.ptp(signal)  # Peak to peak
    features["raw_mean"] = np.mean(signal)
    features["raw_std"] = np.std(signal)
    # Quantiles
    features["raw_q05"] = np.quantile(signal, 0.05)
    features["raw_q95"] = np.quantile(signal, 0.95)
    return features


def process_segment(file_path):
    """
    Loads a sensor file, processes it, and returns a dictionary of features.
    """
    try:
        # Load data
        df = utils.read_sensor_file(file_path)

        # Impute missing values with column mean (Segment-wise)
        df = df.fillna(df.mean())
        # Fallback for all-NaN columns (Cite debug_lesson_1)
        df = df.fillna(0)

        segment_features = {}

        # Iterate over all sensors
        for sensor_col in config.SENSOR_COLS:
            if sensor_col not in df.columns:
                continue

            raw_signal = df[sensor_col].values

            # --- View A: Trend (Savitzky-Golay) ---
            trend_signal = scipy.signal.savgol_filter(
                raw_signal,
                window_length=config.SG_WINDOW,
                polyorder=config.SG_POLYORDER,
            )

            # --- View B: Texture (DWT) ---
            # DWT is applied to the raw signal in extract_wavelet_stats

            # --- View C: Energy/Raw ---
            # Used directly

            # Extract Features
            kinematics = extract_kinematics(trend_signal)
            wavelets = extract_wavelet_stats(raw_signal)
            psd = extract_high_res_psd(raw_signal)
            temporal = extract_temporal_windows(raw_signal)
            basic = extract_basic_stats(raw_signal)

            # Merge and prefix with sensor name
            all_feats = {**kinematics, **wavelets, **psd, **temporal, **basic}

            for k, v in all_feats.items():
                segment_features[f"{sensor_col}_{k}"] = v

        return segment_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return empty dict or handle gracefully;
        # usually better to fail hard in training or return zeros if robust
        return {}


def create_feature_matrix(metadata_df, load_cached_data=True, split_name="train"):
    """
    Generates the feature matrix for the given metadata.
    Handles caching to disk.
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{split_name}_features.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split_name} ({len(metadata_df)} segments)...")

    # 2. Parallel Processing
    # Use joblib to process files in parallel
    results = Parallel(n_jobs=config.LGBM_PARAMS["n_jobs"], verbose=0)(
        delayed(process_segment)(row["file_path"]) for _, row in metadata_df.iterrows()
    )

    # 3. Construct DataFrame
    features_df = pd.DataFrame(results)

    # Attach segment_id
    features_df["segment_id"] = metadata_df["segment_id"].values

    # Attach target if available
    if "time_to_eruption" in metadata_df.columns:
        features_df["time_to_eruption"] = metadata_df["time_to_eruption"].values

    # 4. Save to Cache
    print(f"Saving features to {cache_path}")
    features_df.to_parquet(cache_path, index=False)

    return features_df
