import pandas as pd
from library.config import Config
from library.features import get_dataset as _get_dataset_from_lib


def load_metadata(split):
    """
    Loads the metadata DataFrame for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata DataFrame containing segment_ids and file_paths.
    """
    if split == "train":
        return pd.read_csv(Config.TRAIN_METADATA_PATH)
    elif split == "val":
        return pd.read_csv(Config.VAL_METADATA_PATH)
    elif split == "test":
        return pd.read_csv(Config.TEST_METADATA_PATH)
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )


def generate_dataset(split, load_cached_data=True, debug=Config.DEBUG):
    """
    Generates the dataset (features and targets) for a specific split.
    Uses multiprocessing to process segments and handles caching via the library module.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to load features from cache if available.
        debug (bool): If True, processes only a small subset of the data for debugging.
                      Defaults to Config.DEBUG.

    Returns:
        tuple: (X, y)
            X (pd.DataFrame): Feature matrix. Includes 'segment_id' column.
            y (pd.Series or None): Target values ('time_to_eruption'). None for 'test' split.
    """
    # Update Config.DEBUG based on the function argument to control dataset size
    # This ensures the underlying library function respects the requested debug state
    previous_debug_state = Config.DEBUG
    Config.DEBUG = debug

    try:
        # Identify the correct metadata path based on the split
        if split == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
            )

        # Retrieve the dataframe using the library function
        # This handles checking cache, computing features in parallel, and saving to cache
        df = _get_dataset_from_lib(
            metadata_path, split, load_cached_data=load_cached_data
        )

        # Prepare Features (X) and Target (y)
        if split in ["train", "val"]:
            # Validate that the target column exists
            if "time_to_eruption" not in df.columns:
                raise KeyError(
                    f"'time_to_eruption' column missing in {split} dataset features."
                )

            y = df["time_to_eruption"]
            # We keep segment_id in X as it is useful for tracking,
            # but we drop the target variable from X.
            X = df.drop(columns=["time_to_eruption"])
        else:
            # For the test set, there is no target variable
            y = None
            X = df

        return X, y

    finally:
        # Restore the original Config state to avoid side effects
        Config.DEBUG = previous_debug_state
