import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis, entropy
from joblib import Parallel, delayed
import library.config as config

# ==========================================
# Signal Processing Functions
# ==========================================


def apply_savitzky_golay(x):
    """
    Applies Savitzky-Golay filter to extract trend.
    """
    return savgol_filter(
        x, window_length=config.SG_WINDOW, polyorder=config.SG_POLYORDER
    )


def compute_welch_psd(x, fs=100):
    """
    Computes PSD using Welch's method and integrates power over specific bands.
    """
    f, Pxx = welch(x, fs=fs, nperseg=config.PSD_NPERSEG)

    features = {}
    for band_name, (low, high) in config.PSD_BANDS.items():
        # Find indices corresponding to the frequency band
        idx = np.logical_and(f >= low, f <= high)
        # Integrate power (approximate via sum)
        power = np.sum(Pxx[idx])
        features[f"psd_{band_name}"] = power

    return features


def compute_stats(x, prefix=""):
    """
    Computes basic statistics: Mean, Std, Skew, Kurtosis.
    Handles potential NaNs or constant arrays for Skew/Kurtosis.
    """
    if len(x) == 0:
        return {
            f"{prefix}mean": 0,
            f"{prefix}std": 0,
            f"{prefix}skew": 0,
            f"{prefix}kurt": 0,
        }

    m = np.mean(x)
    s = np.std(x)

    # Skew and Kurtosis can be NaN if std is 0
    if s < 1e-9:
        sk = 0.0
        ku = 0.0
    else:
        sk = skew(x)
        ku = kurtosis(x)

    return {
        f"{prefix}mean": m,
        f"{prefix}std": s,
        f"{prefix}skew": sk,
        f"{prefix}kurt": ku,
    }


def calculate_entropy(x):
    """
    Calculates Shannon entropy of the absolute value distribution (energy distribution).
    """
    if len(x) == 0:
        return 0.0

    # Use absolute values to form a probability distribution
    p = np.abs(x)
    total_energy = np.sum(p)

    if total_energy < 1e-9:
        return 0.0

    p = p / total_energy
    return entropy(p)


# ==========================================
# Feature Extraction Logic
# ==========================================


def extract_features_for_sensor(x, sensor_name):
    """
    Extracts all features for a single sensor array `x`.
    """
    feat_dict = {}

    # 1. Decompositions
    # View C: Raw
    raw = x

    # View A: Trend
    trend = apply_savitzky_golay(raw)

    # Residuals
    residuals = raw - trend

    # 2. Feature Extraction

    # --- From View A (Trend Shape) ---
    # Dense Quantiles
    q_values = np.quantile(trend, config.TREND_QUANTILES)
    for q, val in zip(config.TREND_QUANTILES, q_values):
        feat_dict[f"{sensor_name}_trend_q{int(q*100)}"] = val

    # --- From View A (Trend Kinematics) ---
    # Velocity (1st derivative)
    vel = np.diff(trend)
    feat_dict.update(compute_stats(vel, prefix=f"{sensor_name}_trend_vel_"))

    # Acceleration (2nd derivative)
    acc = np.diff(vel)
    feat_dict.update(compute_stats(acc, prefix=f"{sensor_name}_trend_acc_"))

    # --- From View B (Texture/Residuals) ---
    # Cite solution_lesson_node_00049
    # Extract moments from the high-frequency component
    feat_dict.update(compute_stats(residuals, prefix=f"{sensor_name}_resid_"))
    feat_dict[f"{sensor_name}_resid_rms"] = np.sqrt(np.mean(residuals**2))

    # --- From View C (Raw Intensity) ---
    feat_dict[f"{sensor_name}_raw_min"] = np.min(raw)
    feat_dict[f"{sensor_name}_raw_max"] = np.max(raw)
    feat_dict[f"{sensor_name}_raw_ptp"] = np.ptp(raw)

    # --- From View C (Spectral) ---
    psd_feats = compute_welch_psd(raw)
    for k, v in psd_feats.items():
        feat_dict[f"{sensor_name}_{k}"] = v

    # --- From View C (Temporal Profiling) ---
    # Split into N windows
    windows = np.array_split(raw, config.N_TEMPORAL_WINDOWS)
    rms_vals = []
    mean_vals = []

    for i, w in enumerate(windows):
        if len(w) > 0:
            w_rms = np.sqrt(np.mean(w**2))
            w_mean = np.mean(w)
        else:
            w_rms = 0.0
            w_mean = 0.0

        rms_vals.append(w_rms)
        mean_vals.append(w_mean)

    # Cite solution_lesson_node_00050
    # Aggregate window statistics to ensure shift invariance
    # Instead of win0, win1, etc., we describe the distribution of window stats
    feat_dict[f"{sensor_name}_win_rms_mean"] = np.mean(rms_vals)
    feat_dict[f"{sensor_name}_win_rms_std"] = np.std(rms_vals)
    feat_dict[f"{sensor_name}_win_rms_min"] = np.min(rms_vals)
    feat_dict[f"{sensor_name}_win_rms_max"] = np.max(rms_vals)
    feat_dict[f"{sensor_name}_win_rms_range"] = np.ptp(rms_vals)

    feat_dict[f"{sensor_name}_win_mean_mean"] = np.mean(mean_vals)
    feat_dict[f"{sensor_name}_win_mean_std"] = np.std(mean_vals)

    return feat_dict


def process_segment(meta_row, input_dir):
    """
    Worker function to process a single CSV file.
    """
    segment_id = meta_row["segment_id"]
    file_path = os.path.join(input_dir, meta_row["file_path"])

    try:
        # Load data
        # Using float32 to match dataset description note about loading
        df = pd.read_csv(file_path, dtype="float32")

        # Imputation: Fill missing values with column mean
        df = df.fillna(df.mean())

        # Check if we have all 10 sensors
        sensor_cols = [f"sensor_{i}" for i in range(1, 11)]

        features = {}
        features["segment_id"] = int(segment_id)

        # Extract features for each sensor
        for col in sensor_cols:
            if col in df.columns:
                sensor_data = df[col].values
                sensor_feats = extract_features_for_sensor(sensor_data, col)
                features.update(sensor_feats)
            else:
                # Handle missing sensor column if necessary (unlikely based on description)
                pass

        # Add target if available
        if "time_to_eruption" in meta_row:
            features["time_to_eruption"] = meta_row["time_to_eruption"]

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==========================================
# Main Dataset Creation
# ==========================================


def create_dataset(metadata_path, output_name, load_cached_data=True, debug_size=None):
    """
    Generates or loads the dataset.

    Args:
        metadata_path: Path to the metadata CSV (train.csv, val.csv, or test.csv).
        output_name: Name for the cached file (e.g., 'train_features').
        load_cached_data: Whether to try loading from cache.
        debug_size: If set, limits the number of files processed.

    Returns:
        pd.DataFrame containing features and target (if applicable).
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{output_name}.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {output_name}...")

    # 2. Load Metadata
    meta_df = pd.read_csv(metadata_path)

    # Debugging / Sampling
    if debug_size is not None:
        print(f"Debug Mode: Sampling {debug_size} files.")
        meta_df = meta_df.iloc[:debug_size]

    # 3. Parallel Processing
    # Convert dataframe to list of dicts for iteration
    meta_records = meta_df.to_dict("records")

    results = Parallel(n_jobs=config.N_JOBS, verbose=1)(
        delayed(process_segment)(row, config.INPUT_DIR) for row in meta_records
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # 4. Construct DataFrame
    feature_df = pd.DataFrame(results)

    # Ensure segment_id is int
    if "segment_id" in feature_df.columns:
        feature_df["segment_id"] = feature_df["segment_id"].astype(int)

    # 5. Save Cache
    print(f"Saving features to {cache_path}...")
    feature_df.to_parquet(cache_path, index=False)

    return feature_df


def get_train_data(load_cached_data=True, debug_size=None):
    return create_dataset(
        os.path.join(config.METADATA_DIR, "train.csv"),
        "train_features",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )


def get_val_data(load_cached_data=True, debug_size=None):
    return create_dataset(
        os.path.join(config.METADATA_DIR, "val.csv"),
        "val_features",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )


def get_test_data(load_cached_data=True, debug_size=None):
    return create_dataset(
        os.path.join(config.METADATA_DIR, "test.csv"),
        "test_features",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
