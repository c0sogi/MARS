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


def compute_spectral_features(vals, sensor_name):
    """
    Computes frequency domain features using FFT.
    """
    # Remove DC component (mean) to focus on variance
    vals_centered = vals - np.mean(vals)

    # Compute Real FFT and get magnitude
    fft_vals = np.abs(np.fft.rfft(vals_centered))

    # Spectral Statistics
    feat = {
        f"{sensor_name}_fft_mean": np.mean(fft_vals),
        f"{sensor_name}_fft_std": np.std(fft_vals),
        f"{sensor_name}_fft_max": np.max(fft_vals),
        f"{sensor_name}_fft_q25": np.percentile(fft_vals, 25),
        f"{sensor_name}_fft_q75": np.percentile(fft_vals, 75),
        f"{sensor_name}_peak_freq": np.argmax(fft_vals),  # Index of dominant frequency
    }
    return feat


def compute_rolling_features(vals, sensor_name):
    """
    Computes features over windows to capture temporal evolution.
    """
    # Split signal into 4 non-overlapping windows
    windows = np.array_split(vals, 4)
    feat = {}
    means = []
    stds = []

    for i, w in enumerate(windows):
        m = np.mean(w)
        s = np.std(w)
        means.append(m)
        stds.append(s)
        feat[f"{sensor_name}_win{i}_mean"] = m
        feat[f"{sensor_name}_win{i}_std"] = s

    # Trend features (Last Window - First Window)
    feat[f"{sensor_name}_trend_mean"] = means[-1] - means[0]
    feat[f"{sensor_name}_trend_std"] = stds[-1] - stds[0]

    return feat


def compute_sensor_stats(vals, sensor_name):
    """
    Computes statistical features for a single sensor series (NumPy array).
    """
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
    }

    return features


def extract_segment_features(df):
    """
    Extracts features for all sensors in a segment dataframe.
    """
    all_features = {}
    for col in SENSOR_COLS:
        if col in df.columns:
            # Preprocessing: Handle NaNs and convert to NumPy array once
            series = df[col]
            if series.isnull().any():
                series = series.fillna(series.mean())

            vals = series.values
            if len(vals) == 0:
                vals = np.array([0.0])

            # 1. Time Domain Features
            all_features.update(compute_sensor_stats(vals, col))

            # 2. Frequency Domain Features (Cite solution_lesson_node_00005)
            all_features.update(compute_spectral_features(vals, col))

            # 3. Temporal/Windowed Features (Cite solution_lesson_node_00005)
            all_features.update(compute_rolling_features(vals, col))

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
