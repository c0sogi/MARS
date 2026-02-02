import os
import glob
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed

from library.config import Config
from library.utils import log_message

# ==========================================
# Feature Computation Helper Functions
# ==========================================


def compute_spectral_features(signal, fs=Config.SAMPLING_RATE):
    """
    Computes PSD using Welch's method and integrates power over specific bands.
    Uses a high nperseg to ensure resolution in low frequencies.
    """
    # Compute PSD
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=Config.WELCH_NPERSEG)

    features = {}

    # Integrate power over bands
    # dx is the frequency resolution
    dx = freqs[1] - freqs[0]

    for band_name, (low_f, high_f) in Config.FREQ_BANDS.items():
        # Find indices corresponding to the band
        idx = np.logical_and(freqs >= low_f, freqs <= high_f)
        # Integrate PSD (simple rectangle rule approximation via sum * dx)
        band_power = np.sum(psd[idx]) * dx
        features[f"spec_power_{band_name}"] = band_power

    return features


def compute_kinematic_features(trend_signal):
    """
    Computes derivatives and moments from the trend signal.
    """
    # Velocity (1st derivative)
    velocity = np.gradient(trend_signal)
    # Acceleration (2nd derivative)
    acceleration = np.gradient(velocity)

    features = {}

    def safe_stats(signal, prefix):
        features[f"{prefix}_mean"] = np.mean(signal)
        std = np.std(signal)
        features[f"{prefix}_std"] = std

        if std < 1e-9:
            features[f"{prefix}_skew"] = 0.0
            features[f"{prefix}_kurt"] = 0.0
        else:
            features[f"{prefix}_skew"] = scipy.stats.skew(signal)
            features[f"{prefix}_kurt"] = scipy.stats.kurtosis(signal)

    # Moments on Trend
    safe_stats(trend_signal, "trend")

    # Moments on Velocity
    safe_stats(velocity, "velo")

    # Moments on Acceleration
    features["acc_mean"] = np.mean(acceleration)
    features["acc_std"] = np.std(acceleration)
    features["acc_max"] = np.max(np.abs(acceleration))  # Peak acceleration magnitude

    return features


def compute_hierarchical_volatility(signal):
    """
    Divides signal into windows, computes stats per window,
    then computes stats of those stats.
    """
    # Truncate to fit windows exactly
    n_windows = Config.VOLATILITY_NUM_WINDOWS
    samples_per_window = len(signal) // n_windows
    trunc_len = n_windows * samples_per_window

    reshaped_sig = signal[:trunc_len].reshape(n_windows, samples_per_window)

    # Window-level statistics
    win_means = np.mean(reshaped_sig, axis=1)
    win_rms = np.sqrt(np.mean(reshaped_sig**2, axis=1))

    features = {}

    # Aggregate statistics
    features["vol_std_of_means"] = np.std(win_means)
    features["vol_mean_of_rms"] = np.mean(win_rms)
    features["vol_range_of_rms"] = np.ptp(win_rms)
    features["vol_std_of_rms"] = np.std(win_rms)

    return features


def compute_intensity_features(signal):
    """
    Basic intensity metrics from raw signal.
    """
    return {
        "raw_min": np.min(signal),
        "raw_max": np.max(signal),
        "raw_ptp": np.ptp(signal),
    }


# ==========================================
# Main Processing Logic
# ==========================================


def process_segment(file_path, segment_id):
    """
    Loads a single sensor file, imputes data, and computes features for all 10 sensors.
    Returns a dictionary (row) of features.
    """
    try:
        # Load data
        df = pd.read_csv(file_path, dtype="float32")

        # Imputation: Fill NaNs with column mean
        df = df.fillna(df.mean())

        # If any NaNs remain (e.g., all column was NaN), fill with 0
        df = df.fillna(0)

        segment_features = {"segment_id": segment_id}

        sensor_cols = [c for c in df.columns if "sensor" in c]

        for sensor in sensor_cols:
            raw_sig = df[sensor].values

            # --- Decomposition ---

            # 1. Trend (View A)
            trend_sig = scipy.signal.savgol_filter(
                raw_sig,
                window_length=Config.SG_WINDOW_SIZE,
                polyorder=Config.SG_POLY_ORDER,
            )

            # 2. Texture/Residual (View B)
            texture_sig = raw_sig - trend_sig

            # --- Feature Extraction ---

            # From Trend (Kinematics)
            kinematic_feats = compute_kinematic_features(trend_sig)
            for k, v in kinematic_feats.items():
                segment_features[f"{sensor}_{k}"] = v

            # From Raw (Spectral)
            spectral_feats = compute_spectral_features(raw_sig)
            for k, v in spectral_feats.items():
                segment_features[f"{sensor}_{k}"] = v

            # From Raw (Hierarchical Volatility)
            vol_feats = compute_hierarchical_volatility(raw_sig)
            for k, v in vol_feats.items():
                segment_features[f"{sensor}_{k}"] = v

            # From Raw (Intensity)
            intensity_feats = compute_intensity_features(raw_sig)
            for k, v in intensity_feats.items():
                segment_features[f"{sensor}_{k}"] = v

        return segment_features

    except Exception as e:
        log_message(f"Error processing {file_path}: {e}")
        return None


def get_dataset(metadata_path, dataset_name, load_cached_data=True):
    """
    Orchestrates the loading of features.
    Checks cache first. If not found or forced reload, computes features.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        dataset_name (str): Name identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Feature matrix including segment_id and targets (if available).
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_features.parquet")

    # Load Metadata first to determine expected size
    meta_df = pd.read_csv(metadata_path)

    # Debugging: Sample subset if configured
    if Config.DEBUG:
        log_message(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} files.")
        meta_df = meta_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    expected_len = len(meta_df)

    # 1. Try Loading from Cache with Validation
    if load_cached_data and os.path.exists(cache_path):
        log_message(f"Checking cache: {cache_path}")
        try:
            cached_df = pd.read_parquet(cache_path)
            if len(cached_df) == expected_len:
                log_message(
                    f"Cache valid (Rows: {len(cached_df)}). Loading {dataset_name} features."
                )
                return cached_df
            else:
                log_message(
                    f"Cache invalid (Expected {expected_len} rows, found {len(cached_df)}). Regenerating..."
                )
        except Exception as e:
            log_message(f"Error reading cache: {e}. Regenerating...")

    # 2. Compute from Scratch
    log_message(f"Generating {dataset_name} features from scratch...")

    # Prepare arguments for parallel execution
    # Construct full paths. Metadata contains relative paths.
    file_paths = [os.path.join(Config.INPUT_DIR, p) for p in meta_df["file_path"]]
    segment_ids = meta_df["segment_id"].values

    # Run Parallel Feature Extraction
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_segment)(fp, sid) for fp, sid in zip(file_paths, segment_ids)
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # Create DataFrame
    feature_df = pd.DataFrame(results)

    # Merge with Target if available (train/val sets)
    if "time_to_eruption" in meta_df.columns:
        feature_df = feature_df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 3. Save to Cache
    log_message(f"Saving {dataset_name} features to {cache_path}")
    feature_df.to_parquet(cache_path, index=False)

    return feature_df
