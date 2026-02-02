import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from library.config import Config


def compute_sensor_stats(series):
    """
    Computes statistical features for a single sensor series.

    Args:
        series (pd.Series): Raw sensor readings.

    Returns:
        dict: Dictionary containing statistical features.
    """
    # Remove NaNs for statistical calculation
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q95": 0.0,
        }

    # Basic Stats
    mean_val = np.mean(clean_series)
    std_val = np.std(clean_series)
    min_val = np.min(clean_series)
    max_val = np.max(clean_series)

    # Shape Stats
    skew_val = skew(clean_series)
    kurt_val = kurtosis(clean_series)

    # Quantiles
    quantiles = np.quantile(clean_series, [0.05, 0.25, 0.50, 0.75, 0.95])

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "skew": skew_val,
        "kurt": kurt_val,
        "q05": quantiles[0],
        "q25": quantiles[1],
        "q50": quantiles[2],
        "q75": quantiles[3],
        "q95": quantiles[4],
    }


def process_segment(file_path):
    """
    Loads a segment CSV and extracts features for all sensors.

    Args:
        file_path (str): Full path to the segment CSV file.

    Returns:
        dict: Flattened dictionary of features for the segment.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Return a zero-vector if file is missing (should not happen based on checks)
        # Assuming 10 sensors, ~12 features each
        return {}

    features = {}

    for i in range(1, 11):
        sensor_col = f"sensor_{i}"

        if sensor_col in df.columns:
            series = df[sensor_col]

            # Feature: NaN count
            nan_count = series.isna().sum()
            features[f"{sensor_col}_nan_count"] = nan_count

            # Statistical Features
            stats = compute_sensor_stats(series)
            for stat_name, stat_val in stats.items():
                features[f"{sensor_col}_{stat_name}"] = stat_val
        else:
            # Handle missing sensor column
            features[f"{sensor_col}_nan_count"] = 60001  # Assuming full length missing
            for stat in [
                "mean",
                "std",
                "min",
                "max",
                "skew",
                "kurt",
                "q05",
                "q25",
                "q50",
                "q75",
                "q95",
            ]:
                features[f"{sensor_col}_{stat}"] = 0.0

    return features


def extract_features(metadata_path, cache_path, load_cached_data=True, debug_size=None):
    """
    Main function to extract features for a dataset defined by metadata.
    Handles caching to avoid re-processing.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        cache_path (str): Path where the processed parquet file should be stored/loaded.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug_size (int, optional): If set, only process this many rows for debugging.

    Returns:
        pd.DataFrame: DataFrame containing segment_id, features, and target (if available).
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing features from metadata: {metadata_path}")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    if debug_size is not None:
        df_meta = df_meta.head(debug_size)
        print(f"Debug mode: Processing first {len(df_meta)} segments.")

    # 3. Process Each Segment
    feature_list = []

    # Iterate over metadata rows
    # Using simple loop to avoid multiprocessing complexity within this module,
    # relying on Config.NUM_WORKERS in the training loop if needed,
    # but here we do serial processing for simplicity and stability.
    total = len(df_meta)

    for idx, row in df_meta.iterrows():
        segment_id = row["segment_id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Extract features
        segment_feats = process_segment(full_path)
        segment_feats["segment_id"] = segment_id

        # Add target if it exists
        if "time_to_eruption" in row:
            segment_feats["time_to_eruption"] = row["time_to_eruption"]

        feature_list.append(segment_feats)

        pass

    # 4. Create DataFrame
    df_features = pd.DataFrame(feature_list)

    # 5. Save to Cache
    print(f"Saving features to {cache_path}...")
    df_features.to_parquet(cache_path, index=False)

    return df_features
