import os
import numpy as np
import pandas as pd
import pywt
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    WAVELET_TYPE,
    N_TEMPORAL_WINDOWS,
    NUM_WORKERS,
)
from library.preprocessing import preprocess_segment

# ==========================================
# Feature Extraction Helper Functions
# ==========================================


def extract_view1_intensity(series: np.ndarray) -> dict:
    """
    View 1: Raw Intensity (From Stream A)
    Captures signal severity (outliers are signal).
    """
    return {
        "raw_min": np.min(series),
        "raw_max": np.max(series),
        "raw_ptp": np.ptp(series),
    }


def extract_view2_kinematics(series: np.ndarray) -> dict:
    """
    View 2: Robust Kinematics (From Stream B)
    Captures dynamics (velocity/acceleration) from smoothed signal.
    """
    # First Derivative (Velocity)
    vel = np.diff(series)
    # Second Derivative (Acceleration)
    acc = np.diff(vel)

    feats = {}

    # Velocity Features
    feats["kin_vel_mean"] = np.mean(vel)
    feats["kin_vel_std"] = np.std(vel)
    feats["kin_vel_q01"] = np.quantile(vel, 0.01)
    feats["kin_vel_q05"] = np.quantile(vel, 0.05)
    feats["kin_vel_q95"] = np.quantile(vel, 0.95)
    feats["kin_vel_q99"] = np.quantile(vel, 0.99)

    # Acceleration Features
    feats["kin_acc_mean"] = np.mean(acc)
    feats["kin_acc_std"] = np.std(acc)
    feats["kin_acc_q01"] = np.quantile(acc, 0.01)
    feats["kin_acc_q05"] = np.quantile(acc, 0.05)
    feats["kin_acc_q95"] = np.quantile(acc, 0.95)
    feats["kin_acc_q99"] = np.quantile(acc, 0.99)

    return feats


def extract_view3_wavelets(series: np.ndarray) -> dict:
    """
    View 3: Multi-Resolution Texture (From Stream A)
    Uses Discrete Wavelet Transform.
    """
    feats = {}
    try:
        # Decompose using db4, level 4 is reasonable for ~60k samples
        coeffs = pywt.wavedec(series, WAVELET_TYPE, level=4)

        # coeffs[0] is Approximation (cA4)
        # coeffs[1:] are Details (cD4, cD3, cD2, cD1)

        # Approximation features
        feats[f"wav_approx_energy"] = np.sum(np.square(coeffs[0]))
        feats[f"wav_approx_std"] = np.std(coeffs[0])

        # Detail features (only taking the first couple of levels to save space)
        for i, detail in enumerate(coeffs[1:], start=1):
            level_idx = 5 - i  # roughly mapping to level
            feats[f"wav_detail_l{level_idx}_energy"] = np.sum(np.square(detail))
            feats[f"wav_detail_l{level_idx}_std"] = np.std(detail)

    except Exception:
        # Fallback if pywt fails or data issues
        feats["wav_approx_energy"] = 0.0
        feats["wav_approx_std"] = 0.0

    return feats


def extract_view4_spectral(series: np.ndarray) -> dict:
    """
    View 4: Structural Spectral Features (From Stream A)
    Band Power and Spectral Centroid.
    """
    feats = {}

    # Compute PSD using Welch's method
    # fs=100Hz assumed based on typical seismic data (6000 samples/min)
    f, Pxx = signal.welch(series, fs=100, nperseg=1024)

    # Band Power
    # Low (0-5Hz), Mid (5-20Hz), High (20Hz+)
    # Indices based on freq array f
    idx_low = (f >= 0) & (f < 5)
    idx_mid = (f >= 5) & (f < 20)
    idx_high = f >= 20

    feats["spec_band_low"] = np.sum(Pxx[idx_low])
    feats["spec_band_mid"] = np.sum(Pxx[idx_mid])
    feats["spec_band_high"] = np.sum(Pxx[idx_high])

    # Spectral Centroid
    if np.sum(Pxx) > 0:
        feats["spec_centroid"] = np.sum(f * Pxx) / np.sum(Pxx)
    else:
        feats["spec_centroid"] = 0.0

    return feats


def extract_view5_temporal(series: np.ndarray) -> dict:
    """
    View 5: Flattened Temporal Windows (From Stream A)
    Preserves the 'arrow of time'.
    """
    feats = {}

    # Split into non-overlapping windows
    windows = np.array_split(series, N_TEMPORAL_WINDOWS)

    for i, w in enumerate(windows):
        feats[f"temp_w{i}_mean"] = np.mean(w)
        feats[f"temp_w{i}_rms"] = np.sqrt(np.mean(np.square(w)))

    return feats


def process_single_segment(args):
    """
    Worker function to process a single segment file.
    Args:
        args: tuple (segment_id, file_path_relative)
    Returns:
        dict: Extracted features with segment_id
    """
    segment_id, rel_path = args
    full_path = os.path.join(INPUT_DIR, rel_path)

    try:
        # Get Dual Streams
        stream_a, stream_b = preprocess_segment(full_path)

        segment_features = {"segment_id": segment_id}

        # Iterate over all 10 sensors
        for sensor_col in stream_a.columns:
            if not sensor_col.startswith("sensor_"):
                continue

            # Data arrays
            data_a = stream_a[sensor_col].values
            data_b = stream_b[sensor_col].values

            # Extract Views
            # View 1, 3, 4, 5 from Stream A
            v1 = extract_view1_intensity(data_a)
            v3 = extract_view3_wavelets(data_a)
            v4 = extract_view4_spectral(data_a)
            v5 = extract_view5_temporal(data_a)

            # View 2 from Stream B
            v2 = extract_view2_kinematics(data_b)

            # Combine and prefix with sensor name
            all_sensor_feats = {**v1, **v2, **v3, **v4, **v5}

            for fname, fval in all_sensor_feats.items():
                segment_features[f"{sensor_col}_{fname}"] = fval

        return segment_features

    except Exception as e:
        print(f"Error processing segment {segment_id}: {e}")
        return None


# ==========================================
# Main Feature Generation Function
# ==========================================


def generate_features(
    metadata_df: pd.DataFrame, output_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Generates or loads the feature dataset for the given metadata.

    Args:
        metadata_df (pd.DataFrame): Metadata containing segment_id and file_path.
        output_name (str): Name for the cached file (e.g., 'train_features').
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: The feature matrix.
    """
    cache_path = os.path.join(WORKING_DIR, f"{output_name}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(
        f"Generating features for {len(metadata_df)} segments (Output: {output_name})..."
    )

    # Prepare arguments for parallel processing
    tasks = [(row["segment_id"], row["file_path"]) for _, row in metadata_df.iterrows()]

    # 2. Parallel Feature Extraction
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in executor.map(process_single_segment, tasks):
            if res is not None:
                results.append(res)

    # 3. Create DataFrame
    feature_df = pd.DataFrame(results)

    # Ensure segment_id is int
    if "segment_id" in feature_df.columns:
        feature_df["segment_id"] = feature_df["segment_id"].astype(int)

    # 4. Save Cache
    print(f"Saving features to {cache_path}...")
    feature_df.to_parquet(cache_path, index=False)

    return feature_df
