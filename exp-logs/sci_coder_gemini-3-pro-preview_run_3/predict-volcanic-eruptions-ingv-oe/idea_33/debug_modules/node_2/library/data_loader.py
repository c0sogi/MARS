import os
import pandas as pd
from library.config import PATHS
from library.features import generate_features


def load_metadata(split):
    """
    Loads the metadata DataFrame for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe containing segment_id and file_path.
    """
    if split == "train":
        path = PATHS.TRAIN_CSV
    elif split == "val":
        path = PATHS.VAL_CSV
    elif split == "test":
        path = PATHS.TEST_CSV
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def build_dataset(split, load_cached_data=True, debug_size=None):
    """
    Constructs the feature matrix X and target vector y for a given split.
    Utilizes the library.features module for parallel processing and caching.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading features from the cache.
        debug_size (int, optional): If provided, limits the dataset to the first N samples for debugging.

    Returns:
        tuple: (X, y)
            X (pd.DataFrame): The feature matrix (excluding segment_id and target).
            y (pd.Series or None): The target values (time_to_eruption). None for 'test' split.
    """
    # 1. Determine the source metadata file
    if split == "train":
        source_meta_path = PATHS.TRAIN_CSV
    elif split == "val":
        source_meta_path = PATHS.VAL_CSV
    elif split == "test":
        source_meta_path = PATHS.TEST_CSV
    else:
        raise ValueError(f"Invalid split: {split}")

    dataset_name = split
    meta_path_to_use = source_meta_path

    # 2. Handle Debugging (Subsetting)
    # If debug_size is specified, we create a temporary metadata file containing only the subset.
    # This allows generate_features to process only the requested files without modifying its internal logic.
    if debug_size is not None:
        # Load the full metadata
        full_meta_df = pd.read_csv(source_meta_path)

        # Slice to the requested size
        subset_meta_df = full_meta_df.iloc[:debug_size]

        # Define a unique dataset name for the cache to avoid collisions with the full dataset
        dataset_name = f"{split}_debug_{debug_size}"

        # Save the subset metadata to the working directory
        temp_meta_path = os.path.join(PATHS.WORKING_DIR, f"{dataset_name}_metadata.csv")
        subset_meta_df.to_csv(temp_meta_path, index=False)

        # Point the feature generator to this temporary file
        meta_path_to_use = temp_meta_path

    # 3. Generate Features
    # Calls the provided library function which handles:
    # - Caching (checking parquet files)
    # - Parallel processing of sensor files
    # - Feature extraction logic
    df_features = generate_features(
        metadata_path=meta_path_to_use,
        dataset_name=dataset_name,
        load_cached_data=load_cached_data,
    )

    # 4. Format Output
    # Separate the features (X) from the target (y) and identifiers

    # Identify columns to drop from X (metadata and targets)
    cols_to_drop = ["segment_id"]
    target_col = "time_to_eruption"

    if target_col in df_features.columns:
        y = df_features[target_col]
        cols_to_drop.append(target_col)
    else:
        y = None

    # Create Feature Matrix X
    X = df_features.drop(columns=cols_to_drop, errors="ignore")

    return X, y
