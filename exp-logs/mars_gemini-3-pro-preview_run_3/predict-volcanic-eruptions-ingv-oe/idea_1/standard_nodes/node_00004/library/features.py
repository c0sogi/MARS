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
