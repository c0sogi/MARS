import os
import numpy as np
import pandas as pd
from scipy import signal
import joblib
from library.config import INPUT_DIR, WORKING_DIR, SMOOTHING_WINDOW, POLY_ORDER, N_JOBS


def impute_missing(df):
    """
    Imputes missing values in the DataFrame with the column mean to preserve DC offsets.
    """
    return df.fillna(df.mean())


def apply_smoothing(x):
    """
    Applies Savitzky-Golay smoothing to the signal.
    """
    try:
        return signal.savgol_filter(
            x, window_length=SMOOTHING_WINDOW, polyorder=POLY_ORDER
        )
    except ValueError:
        # Fallback if window_length > len(x)
        return x


def extract_raw_extrema(x, col_name):
    """
    Extracts global minimum, maximum, and peak-to-peak range from the raw signal.
    This corresponds to View 1 (Correction) of the strategy.
    """
    return {
        f"{col_name}_raw_min": np.min(x),
        f"{col_name}_raw_max": np.max(x),
        f"{col_name}_raw_ptp": np.ptp(x),
    }


def extract_kinematics(x_smooth, col_name):
    """
    Computes velocity and acceleration from the smoothed signal and extracts statistical features.
    This corresponds to View 2 (Exploitation) of the strategy.
    """
    features = {}

    # Derivatives (Kinematics)
    vel = np.gradient(x_smooth)
    acc = np.gradient(vel)

    # Statistical Descriptors for Smooth, Velocity, Acceleration
    for name, sig in [("smooth", x_smooth), ("vel", vel), ("acc", acc)]:
        features[f"{col_name}_{name}_mean"] = np.mean(sig)
        features[f"{col_name}_{name}_std"] = np.std(sig)
        features[f"{col_name}_{name}_q01"] = np.quantile(sig, 0.01)
        features[f"{col_name}_{name}_q05"] = np.quantile(sig, 0.05)
        features[f"{col_name}_{name}_q95"] = np.quantile(sig, 0.95)
        features[f"{col_name}_{name}_q99"] = np.quantile(sig, 0.99)

    return features


def extract_spectral(x_smooth, col_name):
    """
    Computes Power Spectral Density features using Welch's method.
    This corresponds to View 3 (Structural Spectral Features).
    """
    features = {}
    # Welch's method for Power Spectral Density
    # Assuming fs=100Hz based on 60000 samples in 10 minutes
    f, Pxx = signal.welch(x_smooth, fs=100, nperseg=256)

    features[f"{col_name}_spec_mean"] = np.mean(Pxx)
    features[f"{col_name}_spec_std"] = np.std(Pxx)
    features[f"{col_name}_spec_max"] = np.max(Pxx)
    features[f"{col_name}_spec_peak_freq"] = f[np.argmax(Pxx)]

    # Band Powers (Cite solution_lesson_node_00007)
    # Low: 0.1-5 Hz (Tremors), Mid: 5-15 Hz, High: 15+ Hz
    idx_low = (f >= 0.1) & (f < 5)
    idx_mid = (f >= 5) & (f < 15)
    idx_high = f >= 15

    features[f"{col_name}_spec_power_low"] = np.sum(Pxx[idx_low])
    features[f"{col_name}_spec_power_mid"] = np.sum(Pxx[idx_mid])
    features[f"{col_name}_spec_power_high"] = np.sum(Pxx[idx_high])

    # Spectral Centroid
    sum_Pxx = np.sum(Pxx)
    if sum_Pxx == 0:
        features[f"{col_name}_spec_centroid"] = 0.0
    else:
        features[f"{col_name}_spec_centroid"] = np.sum(f * Pxx) / sum_Pxx

    return features


def extract_temporal_windows(x, col_name, n_windows=10):
    """
    Divides the raw signal into windows and computes RMS and Mean for each.
    This corresponds to View 4 (Flattened Temporal Windows).
    """
    features = {}
    wins = np.array_split(x, n_windows)
    for i, w in enumerate(wins):
        if len(w) == 0:
            features[f"{col_name}_win_{i}_rms"] = 0.0
            features[f"{col_name}_win_{i}_mean"] = 0.0
        else:
            features[f"{col_name}_win_{i}_rms"] = np.sqrt(np.mean(w**2))
            features[f"{col_name}_win_{i}_mean"] = np.mean(w)

    return features


def process_segment(file_path, segment_id):
    """
    Orchestrates feature extraction for a single data segment.
    """
    try:
        full_path = os.path.join(INPUT_DIR, file_path)
        # Load data (float32 to handle NaNs and memory)
        df = pd.read_csv(full_path, dtype="float32")

        # Impute missing values
        df = impute_missing(df)

        features = {}
        features["segment_id"] = int(segment_id)

        # Identify sensor columns
        sensor_cols = [c for c in df.columns if "sensor" in c]

        for col in sensor_cols:
            x_raw = df[col].values

            # --- View 1: Raw Extrema ---
            features.update(extract_raw_extrema(x_raw, col))

            # --- View 2: Smoothed Kinematics ---
            x_smooth = apply_smoothing(x_raw)
            features.update(extract_kinematics(x_smooth, col))

            # --- View 3: Structural Spectral Features ---
            # Using smoothed signal for spectral analysis to reduce noise
            features.update(extract_spectral(x_smooth, col))

            # --- View 4: Flattened Temporal Windows ---
            # Using raw signal to capture total energy in windows
            features.update(extract_temporal_windows(x_raw, col))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_dataset(metadata_df, dataset_name, load_cached=True):
    """
    Process a dataset (train/val/test) with caching mechanism.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    if load_cached and os.path.exists(cache_file):
        print(f"Loading cached features for {dataset_name} from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating features for {dataset_name}...")

    # Parallel processing for efficiency
    results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(process_segment)(row["file_path"], row["segment_id"])
        for _, row in metadata_df.iterrows()
    )

    # Filter out any failed files
    results = [r for r in results if r is not None]

    if not results:
        raise ValueError(
            f"No features were generated for {dataset_name}. Check input data."
        )

    features_df = pd.DataFrame(results)

    # Save to cache
    print(f"Saving features for {dataset_name} to {cache_file}...")
    features_df.to_parquet(cache_file)

    return features_df
