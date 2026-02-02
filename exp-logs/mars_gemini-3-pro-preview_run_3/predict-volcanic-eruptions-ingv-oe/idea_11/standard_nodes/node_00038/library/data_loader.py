import os
import pandas as pd
import numpy as np
import joblib
from library.config import Config
from library.features import extract_sensor_features


def load_sensor_file(file_path):
    """
    Loads a sensor data file, handling missing values via mean imputation.

    Args:
        file_path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: The loaded and imputed sensor data.
    """
    try:
        # Load data using float32 to save memory
        df = pd.read_csv(file_path, dtype="float32")

        # Imputation: Fill missing values with column mean to preserve DC offsets
        # as per the Dual-Stream pipeline requirements.
        df = df.fillna(df.mean())

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def _process_wrapper(row):
    """
    Worker function to process a single segment.

    Args:
        row (pd.Series): A row from the metadata DataFrame containing 'segment_id' and 'file_path'.

    Returns:
        dict: A dictionary containing the segment_id and all extracted features.
    """
    segment_id = int(row["segment_id"])
    rel_path = row["file_path"]
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    # Load and impute data
    df = load_sensor_file(full_path)

    if df.empty:
        return None

    # Initialize feature dictionary
    features = {"segment_id": segment_id}

    # Extract features for each sensor using the library function
    for sensor in Config.SENSORS:
        if sensor in df.columns:
            sensor_data = df[sensor].values
            # extract_sensor_features handles both Stream A (Raw) and Stream B (Smoothed)
            sensor_feats = extract_sensor_features(sensor_data, sensor)
            features.update(sensor_feats)

    return features


def generate_dataset(metadata_path, cache_name, load_cached_data=True, debug=False):
    """
    Generates or loads the feature dataset for a given metadata file.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        cache_name (str): Name to use for the cached parquet file (e.g., 'train_features').
        load_cached_data (bool): If True, attempts to load from cache first.
        debug (bool): If True, limits the dataset to a small number of samples.

    Returns:
        pd.DataFrame: The feature matrix including targets if available.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache path
    cache_filename = f"{cache_name}{'_debug' if debug else ''}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Load metadata first for validation
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Handle Debug Mode
    if debug:
        meta_df = meta_df.head(50)
        print(f"Debug mode: Processing first {len(meta_df)} segments.")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)

        # Validate Cache (Cite debug_lesson_1, debug_lesson_3)
        is_valid = True
        if len(df) != len(meta_df):
            print(
                f"Cache validation failed: Row count mismatch ({len(df)} vs {len(meta_df)})"
            )
            is_valid = False
        elif (
            "time_to_eruption" in meta_df.columns
            and "time_to_eruption" not in df.columns
        ):
            print("Cache validation failed: Missing target column 'time_to_eruption'")
            is_valid = False

        if is_valid:
            return df

        print("Invalidating cache and regenerating...")

    # 2. Generate from Scratch
    print(f"Generating data for {cache_name} (Debug={debug})...")

    # Parallel Feature Extraction
    # Using n_jobs=12 as per compute specifications
    results = joblib.Parallel(n_jobs=12, backend="loky")(
        joblib.delayed(_process_wrapper)(row) for _, row in meta_df.iterrows()
    )

    # Filter out any failed processings (None)
    results = [r for r in results if r is not None]

    if not results:
        raise RuntimeError("No data was processed successfully.")

    feature_df = pd.DataFrame(results)

    # 3. Merge Target Variable
    # If the metadata contains the target, merge it into the feature set
    if "time_to_eruption" in meta_df.columns:
        target_df = meta_df[["segment_id", "time_to_eruption"]]
        feature_df = feature_df.merge(target_df, on="segment_id", how="left")

    # 4. Save to Cache
    print(f"Saving dataset to {cache_path}")
    feature_df.to_parquet(cache_path, index=False)

    return feature_df
