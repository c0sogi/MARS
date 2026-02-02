import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
import pywt
from joblib import Parallel, delayed
from library.config import Config
from library.utils import load_sensor_data

# ==========================================
# Stream Generation
# ==========================================


def apply_smoothing(signal):
    """
    Generates Stream B: Smoothed signal using Savitzky-Golay filter.
    """
    # Ensure window size is odd and <= signal length
    window_length = Config.SG_WINDOW_SIZE
    if window_length % 2 == 0:
        window_length += 1

    if len(signal) < window_length:
        window_length = len(signal) if len(signal) % 2 != 0 else len(signal) - 1
        if window_length < Config.SG_POLY_ORDER + 2:
            return signal  # Return raw if too short to smooth safely

    return savgol_filter(
        signal, window_length=window_length, polyorder=Config.SG_POLY_ORDER
    )


# ==========================================
# Multi-View Feature Extractors
# ==========================================


def get_intensity_features(signal, sensor_idx):
    """
    View 1: Raw Intensity (From Stream A)
    """
    features = {}
    prefix = f"s{sensor_idx}_raw"

    features[f"{prefix}_min"] = np.min(signal)
    features[f"{prefix}_max"] = np.max(signal)
    features[f"{prefix}_ptp"] = np.ptp(signal)

    return features


def get_kinematic_features(smoothed_signal, sensor_idx):
    """
    View 2: Smoothed Kinematics (From Stream B)
    Computes Velocity (1st Deriv) and Acceleration (2nd Deriv).
    """
    features = {}
    prefix = f"s{sensor_idx}"

    # Velocity
    velocity = np.gradient(smoothed_signal)
    features[f"{prefix}_vel_mean"] = np.mean(velocity)
    features[f"{prefix}_vel_std"] = np.std(velocity)
    features[f"{prefix}_vel_q01"] = np.quantile(velocity, 0.01)
    features[f"{prefix}_vel_q05"] = np.quantile(velocity, 0.05)
    features[f"{prefix}_vel_q95"] = np.quantile(velocity, 0.95)
    features[f"{prefix}_vel_q99"] = np.quantile(velocity, 0.99)

    # Acceleration
    acceleration = np.gradient(velocity)
    features[f"{prefix}_acc_mean"] = np.mean(acceleration)
    features[f"{prefix}_acc_std"] = np.std(acceleration)
    features[f"{prefix}_acc_q01"] = np.quantile(acceleration, 0.01)
    features[f"{prefix}_acc_q05"] = np.quantile(acceleration, 0.05)
    features[f"{prefix}_acc_q95"] = np.quantile(acceleration, 0.95)
    features[f"{prefix}_acc_q99"] = np.quantile(acceleration, 0.99)

    return features


def get_temporal_features(signal, sensor_idx):
    """
    View 3: Explicit Temporal Evolution (From Stream A)
    Splits signal into windows and flattens stats to preserve time order.
    """
    features = {}
    prefix = f"s{sensor_idx}"

    # Split into non-overlapping windows
    windows = np.array_split(signal, Config.NUM_TEMPORAL_WINDOWS)

    for i, w in enumerate(windows):
        # RMS
        rms = np.sqrt(np.mean(w**2))
        features[f"{prefix}_w{i}_rms"] = rms
        # Mean
        features[f"{prefix}_w{i}_mean"] = np.mean(w)

    return features


def get_spectral_features(signal, sensor_idx):
    """
    View 4: Structural Spectral Texture (From Stream A)
    Uses DWT and PSD.
    """
    features = {}
    prefix = f"s{sensor_idx}"

    # 1. Discrete Wavelet Transform
    try:
        coeffs = pywt.wavedec(signal, Config.WAVELET_NAME, level=3)
        # Detail coefficients (high freq) and Approx coefficients (low freq)
        # coeffs[0] is approximation, coeffs[1:] are details
        features[f"{prefix}_dwt_a3_std"] = np.std(coeffs[0])
        features[f"{prefix}_dwt_d3_std"] = np.std(coeffs[1])
        features[f"{prefix}_dwt_d2_std"] = np.std(coeffs[2])
        features[f"{prefix}_dwt_d1_std"] = np.std(coeffs[3])

        features[f"{prefix}_dwt_a3_energy"] = np.sum(coeffs[0] ** 2)
        features[f"{prefix}_dwt_d1_energy"] = np.sum(coeffs[3] ** 2)
    except Exception:
        # Fallback if wavelet fails (e.g. signal too short)
        features[f"{prefix}_dwt_a3_std"] = 0
        features[f"{prefix}_dwt_d3_std"] = 0
        features[f"{prefix}_dwt_d2_std"] = 0
        features[f"{prefix}_dwt_d1_std"] = 0
        features[f"{prefix}_dwt_a3_energy"] = 0
        features[f"{prefix}_dwt_d1_energy"] = 0

    # 2. Power Spectral Density (PSD)
    try:
        f, Pxx = welch(signal, nperseg=min(len(signal), 256))

        # Spectral Centroid
        if np.sum(Pxx) > 0:
            centroid = np.sum(f * Pxx) / np.sum(Pxx)
        else:
            centroid = 0
        features[f"{prefix}_spec_centroid"] = centroid

        # Band Power (Low, Mid, High)
        # Split frequency range into 3 bins
        split_idx = len(Pxx) // 3
        features[f"{prefix}_spec_pow_low"] = np.sum(Pxx[:split_idx])
        features[f"{prefix}_spec_pow_mid"] = np.sum(Pxx[split_idx : 2 * split_idx])
        features[f"{prefix}_spec_pow_high"] = np.sum(Pxx[2 * split_idx :])

    except Exception:
        features[f"{prefix}_spec_centroid"] = 0
        features[f"{prefix}_spec_pow_low"] = 0
        features[f"{prefix}_spec_pow_mid"] = 0
        features[f"{prefix}_spec_pow_high"] = 0

    return features


# ==========================================
# Core Extraction Logic
# ==========================================


def extract_segment_features(row):
    """
    Extracts all features for a single segment (all sensors).

    Args:
        row (pd.Series): A row from the metadata dataframe containing 'segment_id' and 'file_path'.

    Returns:
        dict: A dictionary containing all extracted features + segment_id + target.
    """
    segment_id = row["segment_id"]
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    # Initialize result dict
    result = {"segment_id": segment_id}
    if "time_to_eruption" in row:
        result["time_to_eruption"] = row["time_to_eruption"]

    try:
        # Load Data
        df = load_sensor_data(file_path, fill_na=True)

        # Iterate over sensors
        for i in range(1, Config.NUM_SENSORS + 1):
            sensor_col = f"sensor_{i}"
            if sensor_col not in df.columns:
                continue

            # Stream A: Raw
            raw_signal = df[sensor_col].values

            # Stream B: Smoothed
            smoothed_signal = apply_smoothing(raw_signal)

            # View 1: Intensity (Stream A)
            result.update(get_intensity_features(raw_signal, i))

            # View 2: Kinematics (Stream B)
            result.update(get_kinematic_features(smoothed_signal, i))

            # View 3: Temporal (Stream A)
            result.update(get_temporal_features(raw_signal, i))

            # View 4: Spectral (Stream A)
            result.update(get_spectral_features(raw_signal, i))

    except Exception as e:
        print(f"Error processing segment {segment_id}: {e}")
        # In case of error, we might return a partial dict or handle it upstream.
        # For this pipeline, we'll assume data integrity is mostly good based on analysis.

    return result


def process_dataset(metadata_path, output_filename, load_cached_data=True):
    """
    Main entry point to process a dataset (train, val, or test).
    Handles caching to Parquet.

    Args:
        metadata_path (str): Path to the metadata CSV.
        output_filename (str): Name of the output parquet file (e.g., 'train_features.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The feature dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, output_filename)

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing dataset from {metadata_path}...")

    # 2. Load Metadata
    meta_df = pd.read_csv(metadata_path)

    # Debug mode: sample data
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        meta_df = meta_df.sample(
            n=min(len(meta_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        )

    # 3. Parallel Feature Extraction
    # Convert dataframe to list of dicts/series for iteration
    rows = [row for _, row in meta_df.iterrows()]

    feature_dicts = Parallel(n_jobs=Config.N_JOBS, verbose=0)(
        delayed(extract_segment_features)(row) for row in rows
    )

    # 4. Create DataFrame
    feature_df = pd.DataFrame(feature_dicts)

    # 5. Save to Cache
    print(f"Saving features to {cache_path}...")
    feature_df.to_parquet(cache_path, index=False)

    return feature_df
