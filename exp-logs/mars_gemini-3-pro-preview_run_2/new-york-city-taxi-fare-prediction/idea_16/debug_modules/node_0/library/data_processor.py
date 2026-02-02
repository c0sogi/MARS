import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import clamp_coordinates, haversine_distance


def load_data(path):
    """
    Loads data from a parquet file.

    Args:
        path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_parquet(path)


def filter_strict(df):
    """
    Applies strict filtering for the Wisdom Set (Statistics Generation).
    Removes outliers and physics-inconsistent rows based on Config thresholds.
    Ensures that the statistics derived from this set represent the 'True Expected Price'.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    # Ensure necessary columns exist
    required_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "fare_amount",
    ]

    # Drop NaNs
    df_clean = df.dropna(subset=required_cols).copy()

    # Filter by Fare Amount Range (Strict)
    df_clean = df_clean[
        (df_clean["fare_amount"] >= Config.STRICT_MIN_FARE)
        & (df_clean["fare_amount"] <= Config.STRICT_MAX_FARE)
    ]

    # Calculate Distance for Physics Checks
    # Using numpy arrays for speed
    dist = haversine_distance(
        df_clean["pickup_latitude"].values,
        df_clean["pickup_longitude"].values,
        df_clean["dropoff_latitude"].values,
        df_clean["dropoff_longitude"].values,
    )

    # Filter by Minimum Distance (remove static noise)
    mask_dist = dist > Config.STRICT_MIN_DIST_KM
    df_clean = df_clean[mask_dist]
    dist = dist[mask_dist]  # Align distance array with filtered df

    # Filter by Max Fare per KM (remove unrealistic prices)
    # Logic: fare <= max_rate * dist
    mask_physics = df_clean["fare_amount"] <= (Config.STRICT_MAX_FARE_PER_KM * dist)
    df_clean = df_clean[mask_physics]

    return df_clean


def filter_loose(df):
    """
    Applies loose filtering for the Learner Set (Model Training).
    Retains heavy tails (valid high fares) but removes invalid data.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    required_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "fare_amount",
    ]

    # Drop NaNs
    df_clean = df.dropna(subset=required_cols).copy()

    # Filter by Fare Amount Range (Loose)
    df_clean = df_clean[
        (df_clean["fare_amount"] >= Config.LOOSE_MIN_FARE)
        & (df_clean["fare_amount"] <= Config.LOOSE_MAX_FARE)
    ]

    return df_clean


def get_subsample(df, n_samples):
    """
    Randomly samples n_samples from the dataframe using the global seed.

    Args:
        df (pd.DataFrame): Input dataframe.
        n_samples (int): Number of samples to select.

    Returns:
        pd.DataFrame: Subsampled dataframe.
    """
    if len(df) <= n_samples:
        return df
    return df.sample(n=n_samples, random_state=Config.SEED)


def prepare_datasets(load_cached_data=True):
    """
    Orchestrates the data loading, cleaning, and splitting into Wisdom and Learner sets.
    Implements caching mechanism to store processed datasets in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (wisdom_df, learner_df)
    """
    # Define cache paths
    wisdom_path = os.path.join(Config.WORKING_DIR, "wisdom_set.parquet")
    learner_path = os.path.join(Config.WORKING_DIR, "learner_set.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(wisdom_path) and os.path.exists(learner_path):
            print(f"Loading cached datasets from {Config.WORKING_DIR}...")
            wisdom_df = pd.read_parquet(wisdom_path)
            learner_df = pd.read_parquet(learner_path)
            return wisdom_df, learner_df
        else:
            print(
                "Cached datasets not found or reload forced. Processing from scratch..."
            )

    # 2. Process from scratch
    print(f"Loading raw training data from {Config.TRAIN_DATA_PATH}...")
    # Load full training data
    df = load_data(Config.TRAIN_DATA_PATH)

    # Global Sanitization (Clamp Coordinates)
    print("Clamping coordinates to NYC bounding box...")
    df = clamp_coordinates(df)

    # Create Wisdom Set (Strict Filtering)
    print("Generating Wisdom Set (Strict Filtering)...")
    wisdom_df = filter_strict(df)

    # Create Learner Set (Loose Filtering + Subsampling)
    print("Generating Learner Set (Loose Filtering + Subsampling)...")
    learner_full = filter_loose(df)
    learner_df = get_subsample(learner_full, Config.TRAIN_SUBSAMPLE_SIZE)

    # 3. Save to cache
    print(f"Saving processed datasets to {Config.WORKING_DIR}...")
    wisdom_df.to_parquet(wisdom_path, index=False)
    learner_df.to_parquet(learner_path, index=False)

    return wisdom_df, learner_df
