import pandas as pd
from library import config
from library import features


def _split_features_target_ids(df, is_test=False):
    """
    Helper function to separate features, target, and IDs from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing all data.
        is_test (bool): Whether this is the test set (no target column).

    Returns:
        tuple: (X, y, ids) for train/val, or (X, ids) for test.
    """
    # Define columns that are metadata or targets, not input features
    # 'file_path' is used for augmentation but not as a direct feature
    # 'id' is the identifier
    # 'species' is the target class
    exclude_cols = {"id", "species", "file_path"}

    # Identify feature columns: all columns in the dataframe that are not excluded.
    # This automatically includes the 192 original features plus 'area' and 'aspect_ratio'.
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Extract Features
    X = df[feature_cols].copy()

    # Extract IDs
    ids = df["id"].copy()

    if is_test:
        return X, ids
    else:
        # Verify target exists
        if "species" not in df.columns:
            raise ValueError("Column 'species' missing from non-test dataset.")

        # Extract Target
        y = df["species"].copy()
        return X, y, ids


def load_dataset(
    load_cached_data=True,
    debug_mode=config.DEBUG_MODE,
    debug_size=config.DEBUG_SUBSET_SIZE,
):
    """
    Loads the training, validation, and test datasets.
    Delegates augmentation and caching to library.features.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-computed parquet files.
        debug_mode (bool): If True, returns a small subset of the data for debugging.
        debug_size (int): The number of samples to return per split in debug mode.

    Returns:
        tuple: (train_data, val_data, test_data)
               where train_data = (X_train, y_train, train_ids)
                     val_data   = (X_val, y_val, val_ids)
                     test_data  = (X_test, test_ids)
    """

    # Load augmented dataframes using the features library.
    # The features.get_augmented_dataset function handles the logic of checking
    # for a cached parquet file, and if not found, loading metadata, computing
    # geometric features from images, and saving to cache.
    df_train = features.get_augmented_dataset(
        "train", load_cached_data=load_cached_data
    )
    df_val = features.get_augmented_dataset("val", load_cached_data=load_cached_data)
    df_test = features.get_augmented_dataset("test", load_cached_data=load_cached_data)

    # Apply Debugging Subsampling if requested
    if debug_mode:
        print(
            f"Debug Mode: Filtering to common classes and sampling {debug_size} rows."
        )

        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows.
        # We must ensure that the validation set only contains classes present in the training set.

        # 1. Identify common species to ensure overlap potential
        train_species = set(df_train["species"].unique())
        val_species = set(df_val["species"].unique())
        common_species = sorted(list(train_species.intersection(val_species)))

        # Fallback if intersection is empty (unlikely but possible in edge cases)
        if not common_species:
            common_species = sorted(list(train_species))

        # 2. Select a small subset of classes (e.g., top 5) to keep dimensionality manageable
        selected_species = common_species[:5]

        # 3. Filter Train Data first
        df_train = df_train[df_train["species"].isin(selected_species)]
        df_train = df_train.iloc[:debug_size]

        # 4. Determine exactly which species survived the slicing in Train
        # (Slicing might have removed some of the 5 selected species)
        active_train_species = df_train["species"].unique()

        # 5. Filter Val Data to ONLY allow species that are actually in the reduced Train
        df_val = df_val[df_val["species"].isin(active_train_species)]
        df_val = df_val.iloc[:debug_size]

        # 6. Slice Test Data (no labels, so simple slice is fine)
        df_test = df_test.iloc[:debug_size]

    # Structure the data into (X, y, id) tuples
    train_data = _split_features_target_ids(df_train, is_test=False)
    val_data = _split_features_target_ids(df_val, is_test=False)
    test_data = _split_features_target_ids(df_test, is_test=True)

    return train_data, val_data, test_data
