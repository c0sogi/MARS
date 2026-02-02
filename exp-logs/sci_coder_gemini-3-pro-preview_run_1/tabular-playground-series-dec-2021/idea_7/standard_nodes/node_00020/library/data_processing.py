import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_PATH,
    TEST_PATH,
    WORKING_DIR,
    SOIL_COLUMNS,
    WILDERNESS_COLUMNS,
    USE_GEOMETRIC_FEATURES,
    USE_REVERSE_ONE_HOT,
)


class FeatureEngineer:
    """
    Handles feature engineering transformations including Reverse One-Hot Encoding
    and Geometric Feature generation.
    """

    def __init__(self):
        pass

    def reverse_one_hot(self, df):
        """
        Condenses binary Soil_Type and Wilderness_Area columns into dense integer indices.
        Uses the numerically sorted column lists from config to ensure correct mapping.
        """
        # Process Soil Types
        # Filter to columns actually present in the dataframe
        present_soil_cols = [c for c in SOIL_COLUMNS if c in df.columns]
        if present_soil_cols:
            # Create dense index (0 to 39)
            # argmax returns the index of the maximum value (1) along the row
            # We assume rows are mutually exclusive or we take the first occurrence
            df["Soil_Type"] = np.argmax(df[present_soil_cols].values, axis=1)
            # Cite solution_lesson_node_00019: Do not replace One-Hot Encoded features; add them as complementary features.
            # We keep the original binary columns.

        # Process Wilderness Areas
        present_wild_cols = [c for c in WILDERNESS_COLUMNS if c in df.columns]
        if present_wild_cols:
            # Create dense index (0 to 3)
            df["Wilderness_Area"] = np.argmax(df[present_wild_cols].values, axis=1)
            # Cite solution_lesson_node_00019: Keep original binary columns.

        return df

    def add_geometric_features(self, df):
        """
        Generates physics-informed geometric features:
        1. Euclidean Distance to Hydrology
        2. Relative Elevation (Elevation - Vertical Distance to Hydrology)
        3. Cyclic Aspect (Sin/Cos)
        """
        # Euclidean Distance to Hydrology
        if (
            "Horizontal_Distance_To_Hydrology" in df.columns
            and "Vertical_Distance_To_Hydrology" in df.columns
        ):
            df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
                df["Horizontal_Distance_To_Hydrology"] ** 2
                + df["Vertical_Distance_To_Hydrology"] ** 2
            )

        # Relative Elevation
        if "Elevation" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
            df["Elevation_Hydrology_Diff"] = (
                df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
            )

        # Cyclic Aspect
        if "Aspect" in df.columns:
            # Aspect is in degrees, convert to radians
            aspect_rad = np.deg2rad(df["Aspect"])
            df["Aspect_Sin"] = np.sin(aspect_rad)
            df["Aspect_Cos"] = np.cos(aspect_rad)

        return df

    def process(self, df):
        """
        Applies configured feature engineering steps.
        """
        if USE_REVERSE_ONE_HOT:
            df = self.reverse_one_hot(df)

        if USE_GEOMETRIC_FEATURES:
            df = self.add_geometric_features(df)

        return df


def load_and_process(load_cached_data=True, debug_sample_size=None):
    """
    Loads data, applies feature engineering, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
        debug_sample_size (int or None): If set, subsamples the training data for debugging.

    Returns:
        train_df (pd.DataFrame): Processed training data.
        test_df (pd.DataFrame): Processed test data.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache_path = os.path.join(WORKING_DIR, "train_processed.parquet")
    test_cache_path = os.path.join(WORKING_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached processed data from {WORKING_DIR}...")
        train_df = pd.read_parquet(train_cache_path)
        test_df = pd.read_parquet(test_cache_path)

        # Apply debug sampling if requested, even on cached data
        if debug_sample_size is not None:
            print(
                f"Debug mode: Subsampling loaded train data to {debug_sample_size} rows."
            )
            if len(train_df) > debug_sample_size:
                train_df = train_df.iloc[:debug_sample_size]

        return train_df, test_df

    # 2. Process from scratch
    print("Cache not found or reload requested. Processing data from scratch...")

    # Load raw data
    print(f"Reading training data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)

    print(f"Reading test data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)

    # Apply debug sampling on raw data to speed up processing if debugging
    if debug_sample_size is not None:
        print(f"Debug mode: Subsampling raw train data to {debug_sample_size} rows.")
        if len(train_df) > debug_sample_size:
            train_df = train_df.iloc[:debug_sample_size]

    # Initialize Feature Engineer
    fe = FeatureEngineer()

    # Process Train
    print("Applying feature engineering to Training set...")
    train_df = fe.process(train_df)

    # Process Test
    print("Applying feature engineering to Test set...")
    test_df = fe.process(test_df)

    # 3. Save to cache
    print(f"Saving processed data to {WORKING_DIR}...")
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df
