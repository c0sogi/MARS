import os
import numpy as np
import pandas as pd
from library.config import RAW_TRAIN_PATH, RAW_TEST_PATH, WORKING_DIR


def engineer_features(df):
    """
    Applies physics-informed geometric transformations to the dataset.

    Args:
        df (pd.DataFrame): Input dataframe containing raw features.

    Returns:
        pd.DataFrame: Dataframe with added engineered features.
    """
    # Euclidean Distance to Hydrology
    # Combines horizontal and vertical distances into a direct distance metric
    if (
        "Horizontal_Distance_To_Hydrology" in df.columns
        and "Vertical_Distance_To_Hydrology" in df.columns
    ):
        df["Hydro_Euclidean"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

    # Relative Elevation
    # Elevation relative to the nearest water source
    if "Elevation" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
        df["Elev_Relative"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # Cyclic Aspect Encoding
    # Transforms angular Aspect (degrees) into continuous sin/cos components
    if "Aspect" in df.columns:
        # Convert degrees to radians
        aspect_rad = np.radians(df["Aspect"])
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

    # Hillshade Aggregates
    # Captures average solar exposure and daily variation (Cite solution_lesson_node_00007)
    hillshade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    if all(c in df.columns for c in hillshade_cols):
        df["Hillshade_Mean"] = df[hillshade_cols].mean(axis=1)
        df["Hillshade_Range"] = df[hillshade_cols].max(axis=1) - df[hillshade_cols].min(
            axis=1
        )

    # Reverse One-Hot Encoding
    # Condenses sparse binary features into dense categorical indices for efficient tree splits

    # Wilderness Areas
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    if wild_cols:
        # Use argmax to find the active index (0 to N-1)
        # This assumes rows are one-hot or zero (if zero, argmax returns 0, which is fine as a category)
        df["Wilderness_Area_Index"] = np.argmax(df[wild_cols].values, axis=1)

    # Soil Types
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    if soil_cols:
        df["Soil_Type_Index"] = np.argmax(df[soil_cols].values, axis=1)

    return df


def load_and_prepare_data(load_cached_data=True):
    """
    Loads training and test data, applies feature engineering, and prepares arrays for training.
    Implements a caching mechanism using Parquet files to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.

    Returns:
        tuple: (X, y, X_test, test_ids)
            - X (pd.DataFrame): Training features.
            - y (pd.Series): Training targets.
            - X_test (pd.DataFrame): Test features.
            - test_ids (pd.Series): Test IDs for submission.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_X_path = os.path.join(WORKING_DIR, "X_train.parquet")
    cache_y_path = os.path.join(WORKING_DIR, "y_train.parquet")
    cache_X_test_path = os.path.join(WORKING_DIR, "X_test.parquet")
    cache_ids_path = os.path.join(WORKING_DIR, "test_ids.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_X_path)
            and os.path.exists(cache_y_path)
            and os.path.exists(cache_X_test_path)
            and os.path.exists(cache_ids_path)
        ):

            print("Loading processed data from cache...")
            X = pd.read_parquet(cache_X_path)
            # Read y as DataFrame and convert to Series
            y = pd.read_parquet(cache_y_path).iloc[:, 0]
            X_test = pd.read_parquet(cache_X_test_path)
            # Read IDs as DataFrame and convert to Series
            test_ids = pd.read_parquet(cache_ids_path).iloc[:, 0]

            return X, y, X_test, test_ids
        else:
            print("Cache missing or incomplete. Processing data from scratch...")
    else:
        print("Cache loading disabled. Processing data from scratch...")

    # Load raw data
    print(f"Reading training data from {RAW_TRAIN_PATH}...")
    train_df = pd.read_csv(RAW_TRAIN_PATH)

    print(f"Reading test data from {RAW_TEST_PATH}...")
    test_df = pd.read_csv(RAW_TEST_PATH)

    # Apply feature engineering
    print("Applying feature engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Process Training Data
    if "Cover_Type" not in train_df.columns:
        raise ValueError("Target column 'Cover_Type' not found in training data.")

    y = train_df["Cover_Type"]
    X = train_df.drop(columns=["Cover_Type"])

    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    # Process Test Data
    if "Id" not in test_df.columns:
        raise ValueError("Column 'Id' not found in test data.")

    test_ids = test_df["Id"]
    X_test = test_df.drop(columns=["Id"])

    # Save to cache
    print("Saving processed data to cache...")
    X.to_parquet(cache_X_path)
    # Save Series as DataFrame for Parquet compatibility
    y.to_frame(name="Cover_Type").to_parquet(cache_y_path)
    X_test.to_parquet(cache_X_test_path)
    test_ids.to_frame(name="Id").to_parquet(cache_ids_path)

    return X, y, X_test, test_ids
