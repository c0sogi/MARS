import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    SENSOR_COLS,
    SEED,
)

# Set seed for reproducibility
np.random.seed(SEED)


def compute_sensor_stats(sensor_series, sensor_name):
    """
    Computes statistical features for a single sensor series.
    """
    # Impute missing values with mean
    if sensor_series.isnull().any():
        sensor_series = sensor_series.fillna(sensor_series.mean())

    # Edge case: if series is empty or all-NaN even after fillna (e.g. empty file)
    if len(sensor_series) == 0:
        # Return zeros to maintain schema consistency
        vals = np.array([0.0])
    else:
        vals = sensor_series.values

    # Basic Stats
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    var_val = np.var(vals)
    min_val = np.min(vals)
    max_val = np.max(vals)
    range_val = max_val - min_val

    # Quantiles
    q25 = np.percentile(vals, 25)
    median_val = np.median(vals)
    q75 = np.percentile(vals, 75)
    iqr_val = q75 - q25

    # Shape
    skew_val = skew(vals)
    kurt_val = kurtosis(vals)

    # Dynamics (Mean Absolute Difference)
    diff = np.diff(vals)
    mad_diff = np.mean(np.abs(diff)) if len(diff) > 0 else 0.0

    # Frequency Domain Features (Cite solution_lesson_node_00005, solution_lesson_node_00007)
    # FFT
    fft_vals = np.abs(np.fft.rfft(vals))

    if len(fft_vals) > 0:
        fft_mean = np.mean(fft_vals)
        fft_std = np.std(fft_vals)
        fft_q95 = np.percentile(fft_vals, 95)

        # Band Powers (Assuming ~100Hz sampling rate based on 60k samples/10mins)
        # 60k samples -> rfft has ~30k bins.
        # Nyquist = 50Hz. 30k bins -> 50Hz. 1 bin ~= 0.00166 Hz.
        # Low (0.1-3Hz) -> ~60 to ~1800
        # Mid (3-10Hz) -> ~1800 to ~6000
        # High (10-20Hz) -> ~6000 to ~12000
        fft_low = np.sum(fft_vals[60:1800])
        fft_mid = np.sum(fft_vals[1800:6000])
        fft_high = np.sum(fft_vals[6000:12000])
    else:
        fft_mean = fft_std = fft_q95 = fft_low = fft_mid = fft_high = 0.0

    # Windowed Features (Cite solution_lesson_node_00005)
    # Capture temporal evolution by splitting into windows
    n_windows = 10
    # Truncate to multiple of n_windows for simple reshaping
    trim_len = (len(vals) // n_windows) * n_windows
    if trim_len > 0:
        vals_win = vals[:trim_len].reshape(n_windows, -1)
        win_means = np.mean(vals_win, axis=1)
        win_stds = np.std(vals_win, axis=1)

        # Aggregates of window stats
        win_mean_std = np.std(win_means)  # How much the mean shifts
        win_std_mean = np.mean(win_stds)  # Average noise level
        win_std_std = np.std(win_stds)  # Variability of noise level
    else:
        win_mean_std = win_std_mean = win_std_std = 0.0

    features = {
        f"{sensor_name}_mean": mean_val,
        f"{sensor_name}_std": std_val,
        f"{sensor_name}_var": var_val,
        f"{sensor_name}_min": min_val,
        f"{sensor_name}_max": max_val,
        f"{sensor_name}_range": range_val,
        f"{sensor_name}_q25": q25,
        f"{sensor_name}_median": median_val,
        f"{sensor_name}_q75": q75,
        f"{sensor_name}_iqr": iqr_val,
        f"{sensor_name}_skew": skew_val,
        f"{sensor_name}_kurt": kurt_val,
        f"{sensor_name}_mad_diff": mad_diff,
        # New Features
        f"{sensor_name}_fft_mean": fft_mean,
        f"{sensor_name}_fft_std": fft_std,
        f"{sensor_name}_fft_q95": fft_q95,
        f"{sensor_name}_fft_low": fft_low,
        f"{sensor_name}_fft_mid": fft_mid,
        f"{sensor_name}_fft_high": fft_high,
        f"{sensor_name}_win_mean_std": win_mean_std,
        f"{sensor_name}_win_std_mean": win_std_mean,
        f"{sensor_name}_win_std_std": win_std_std,
    }

    return features


def extract_segment_features(df):
    """
    Extracts features for all sensors in a segment dataframe.
    """
    all_features = {}
    for col in SENSOR_COLS:
        if col in df.columns:
            stats = compute_sensor_stats(df[col], col)
            all_features.update(stats)
    return all_features


def process_dataset(meta_path, input_dir, debug_size=None):
    """
    Loads metadata, iterates through files, extracts features, and returns a DataFrame.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    if debug_size is not None:
        meta_df = meta_df.head(debug_size)

    feature_list = []

    # Iterate through each file
    for _, row in meta_df.iterrows():
        segment_id = row["segment_id"]
        file_rel_path = row["file_path"]
        file_full_path = os.path.join(input_dir, file_rel_path)

        try:
            # Load raw data
            # Using float32 to handle NaNs and optimize memory usage
            df = pd.read_csv(file_full_path, dtype="float32")

            # Extract features
            features = extract_segment_features(df)

            # Add metadata
            features["segment_id"] = int(segment_id)
            if "time_to_eruption" in row:
                features["time_to_eruption"] = row["time_to_eruption"]

            feature_list.append(features)

        except Exception as e:
            print(f"Error processing {file_full_path}: {e}")
            continue

    if not feature_list:
        return pd.DataFrame()

    return pd.DataFrame(feature_list)


def generate_features(load_cached_data=True, debug_size=None):
    """
    Main function to generate or load features for train, val, and test sets.
    Implements caching logic using Parquet files.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(TRAIN_FEATURES_PATH), exist_ok=True)

    # --- Train Data ---
    if load_cached_data and os.path.exists(TRAIN_FEATURES_PATH):
        print(f"Loading cached train features from {TRAIN_FEATURES_PATH}...")
        train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    else:
        print("Processing train data...")
        train_df = process_dataset(TRAIN_META_PATH, INPUT_DIR, debug_size=debug_size)
        train_df.to_parquet(TRAIN_FEATURES_PATH, index=False)
        print(f"Saved train features to {TRAIN_FEATURES_PATH}")

    # --- Validation Data ---
    if load_cached_data and os.path.exists(VAL_FEATURES_PATH):
        print(f"Loading cached val features from {VAL_FEATURES_PATH}...")
        val_df = pd.read_parquet(VAL_FEATURES_PATH)
    else:
        print("Processing val data...")
        val_df = process_dataset(VAL_META_PATH, INPUT_DIR, debug_size=debug_size)
        val_df.to_parquet(VAL_FEATURES_PATH, index=False)
        print(f"Saved val features to {VAL_FEATURES_PATH}")

    # --- Test Data ---
    if load_cached_data and os.path.exists(TEST_FEATURES_PATH):
        print(f"Loading cached test features from {TEST_FEATURES_PATH}...")
        test_df = pd.read_parquet(TEST_FEATURES_PATH)
    else:
        print("Processing test data...")
        test_df = process_dataset(TEST_META_PATH, INPUT_DIR, debug_size=debug_size)
        test_df.to_parquet(TEST_FEATURES_PATH, index=False)
        print(f"Saved test features to {TEST_FEATURES_PATH}")

    return train_df, val_df, test_df
