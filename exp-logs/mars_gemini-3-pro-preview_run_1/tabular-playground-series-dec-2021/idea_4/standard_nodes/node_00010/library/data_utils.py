import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import DATA_PATHS, CACHE_DIR, TARGET_COL, PIPELINE_PARAMS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies geometric feature engineering to the dataset.

    Features created:
    - Euclidean_Distance_To_Hydrology
    - Relative_Elevation_Hydrology
    - Aspect_Sin
    - Aspect_Cos
    """
    df = df.copy()

    # Ensure required columns exist
    req_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Elevation",
        "Aspect",
    ]

    if not all(col in df.columns for col in req_cols):
        # If columns are missing (e.g. already processed or different schema), return as is
        return df

    # Euclidean Distance to Hydrology
    # sqrt(h_dist^2 + v_dist^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Relative Elevation
    # Elevation - Vertical_Distance_To_Hydrology gives the elevation at the hydrology point
    df["Relative_Elevation_Hydrology"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # Cyclic Aspect Encoding
    # Aspect is in degrees (0-360). Convert to radians first.
    aspect_rad = np.radians(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    return df


def load_dataset(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.
    Implements caching using Parquet files in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    train_cache = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Loading datasets from raw metadata...")
    # Load raw data
    train_df = pd.read_csv(DATA_PATHS["train_path"])
    val_df = pd.read_csv(DATA_PATHS["val_path"])
    test_df = pd.read_csv(DATA_PATHS["test_path"])

    # Apply feature engineering
    if PIPELINE_PARAMS.get("use_geometry_features", True):
        print("Applying feature engineering...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

    # Save to cache
    print(f"Saving processed datasets to {CACHE_DIR}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_stratified_folds(df: pd.DataFrame, n_folds: int = 5, random_state: int = 42):
    """
    Generates stratified K-Fold indices.

    Args:
        df (pd.DataFrame): Training dataframe containing the target column.
        n_folds (int): Number of folds.
        random_state (int): Seed for reproducibility.

    Returns:
        generator: Yields (train_index, val_index) tuples.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return skf.split(df, df[TARGET_COL])


def create_augmented_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_probs: np.ndarray,
    threshold: float = 0.99,
) -> pd.DataFrame:
    """
    Creates an augmented training set by merging the original train set
    with high-confidence pseudo-labeled test samples.

    Args:
        train_df (pd.DataFrame): Original training data.
        test_df (pd.DataFrame): Test data (features only).
        test_probs (np.ndarray): Predicted probabilities for the test set (N_samples, N_classes).
        threshold (float): Confidence threshold for pseudo-labeling.

    Returns:
        pd.DataFrame: Augmented training dataframe.
    """
    print(f"Generating pseudo-labels with threshold {threshold}...")

    # Identify the classes present in the training data to map indices correctly
    # XGBoost/Sklearn usually output probs sorted by class label
    unique_classes = sorted(train_df[TARGET_COL].unique())

    # Get max probability and the corresponding class index for each sample
    max_probs = np.max(test_probs, axis=1)
    pred_indices = np.argmax(test_probs, axis=1)

    # Map indices to actual class labels
    pred_labels = np.array([unique_classes[i] for i in pred_indices])

    # Filter for high confidence
    high_conf_mask = max_probs >= threshold

    if np.sum(high_conf_mask) == 0:
        print(
            "No samples met the pseudo-labeling threshold. Returning original train set."
        )
        return train_df.copy()

    # Create pseudo-labeled dataframe
    pseudo_df = test_df.loc[high_conf_mask].copy()
    pseudo_df[TARGET_COL] = pred_labels[high_conf_mask]

    print(f"Added {len(pseudo_df)} pseudo-labeled samples to training data.")

    # Concatenate with original training data
    # Ensure columns match (test_df might lack Target initially, but we just added it)
    # Align columns just in case order differs
    pseudo_df = pseudo_df[train_df.columns]

    augmented_train_df = pd.concat([train_df, pseudo_df], axis=0, ignore_index=True)

    return augmented_train_df
