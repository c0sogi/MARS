import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config, data_loader


class GroupWisePreprocessor:
    """
    Manages separate PowerTransformer and StandardScaler instances for each
    semantic feature group (margin, shape, texture).

    Attributes:
        transformers (dict): Stores fitted PowerTransformer instances keyed by group.
        scalers (dict): Stores fitted StandardScaler instances keyed by group.
        groups (list): List of semantic feature prefixes defined in config.
    """

    def __init__(self):
        self.transformers = {}
        self.scalers = {}
        self.groups = config.FEATURE_PREFIXES  # e.g., ['margin', 'shape', 'texture']

    def fit(self, df_train):
        """
        Fits transformers on the training data for each group independently.

        Args:
            df_train (pd.DataFrame): Training data containing all features.

        Returns:
            self
        """
        # Split features into groups (margin, shape, texture)
        feature_groups = data_loader.split_features_by_group(df_train)

        for group in self.groups:
            if group not in feature_groups:
                continue

            # Extract data and ensure float64 precision
            X_g = feature_groups[group].values.astype(config.FLOAT_PRECISION)

            # 1. Yeo-Johnson Power Transformation (standardize=False)
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            X_g_pt = pt.fit_transform(X_g)

            # 2. Standard Scaling
            ss = StandardScaler()
            ss.fit(X_g_pt)

            # Store fitted transformers
            self.transformers[group] = pt
            self.scalers[group] = ss

        return self

    def transform(self, df):
        """
        Applies the fitted transformers to the dataframe.

        Args:
            df (pd.DataFrame): Data to transform (train, val, or test).

        Returns:
            dict: A dictionary mapping group names to transformed numpy arrays (float64).
        """
        feature_groups = data_loader.split_features_by_group(df)
        transformed_data = {}

        for group in self.groups:
            if group not in feature_groups:
                continue

            if group not in self.transformers:
                raise ValueError(
                    f"Transformer for group '{group}' has not been fitted."
                )

            X_g = feature_groups[group].values.astype(config.FLOAT_PRECISION)

            # Retrieve transformers
            pt = self.transformers[group]
            ss = self.scalers[group]

            # Apply transformations
            X_g_pt = pt.transform(X_g)
            X_g_ss = ss.transform(X_g_pt)

            # Store result ensuring precision
            transformed_data[group] = X_g_ss.astype(config.FLOAT_PRECISION)

        return transformed_data


def get_preprocessed_data(load_cached_data=True, debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Orchestrates loading, preprocessing, and caching of data.

    Checks for existing .npy files in the cache directory. If found and load_cached_data is True,
    loads them. Otherwise, loads raw data, fits the GroupWisePreprocessor on train,
    transforms all sets, and saves the results to cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_size (int or None): Number of samples to use for debugging.

    Returns:
        tuple: (
            X_train_dict, y_train,
            X_val_dict, y_val,
            X_test_dict, ids_test
        )
        where X_*_dict is { 'margin': np.array, 'shape': np.array, 'texture': np.array }
    """
    # Define cache filenames helper
    splits = ["train", "val", "test"]
    groups = config.FEATURE_PREFIXES

    def get_X_path(split, group):
        return os.path.join(config.CACHE_DIR, f"X_{split}_{group}.npy")

    def get_y_path(split):
        return os.path.join(config.CACHE_DIR, f"y_{split}.npy")

    def get_id_path(split):
        return os.path.join(config.CACHE_DIR, f"ids_{split}.npy")

    # Check cache integrity
    cache_complete = True
    for split in splits:
        # Check feature matrices
        for group in groups:
            if not os.path.exists(get_X_path(split, group)):
                cache_complete = False
                break

        # Check targets/IDs
        if split == "test":
            if not os.path.exists(get_id_path(split)):
                cache_complete = False
        else:
            if not os.path.exists(get_y_path(split)):
                cache_complete = False

    # Load from cache if valid
    if load_cached_data and cache_complete:
        print("Loading preprocessed data from cache...")
        data_out = {}

        # Load X dictionaries
        for split in splits:
            group_dict = {}
            for group in groups:
                path = get_X_path(split, group)
                group_dict[group] = np.load(path)
            data_out[f"X_{split}"] = group_dict

        # Load y and ids
        # allow_pickle=True is required for string arrays (species names)
        data_out["y_train"] = np.load(get_y_path("train"), allow_pickle=True)
        data_out["y_val"] = np.load(get_y_path("val"), allow_pickle=True)
        data_out["ids_test"] = np.load(get_id_path("test"))

        return (
            data_out["X_train"],
            data_out["y_train"],
            data_out["X_val"],
            data_out["y_val"],
            data_out["X_test"],
            data_out["ids_test"],
        )

    # Compute from scratch
    print("Computing preprocessed data from scratch...")

    # Load raw data via data_loader
    df_train, df_val, df_test = data_loader.load_datasets(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # Extract targets and IDs
    y_train = df_train[config.TARGET_COLUMN].values
    y_val = df_val[config.TARGET_COLUMN].values
    ids_test = df_test[config.ID_COLUMN].values

    # Initialize and fit preprocessor
    preprocessor = GroupWisePreprocessor()
    preprocessor.fit(df_train)

    # Transform all datasets
    X_train_dict = preprocessor.transform(df_train)
    X_val_dict = preprocessor.transform(df_val)
    X_test_dict = preprocessor.transform(df_test)

    # Save to cache
    print(f"Saving preprocessed data to {config.CACHE_DIR}...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Save Feature Matrices
    for group in groups:
        np.save(get_X_path("train", group), X_train_dict[group])
        np.save(get_X_path("val", group), X_val_dict[group])
        np.save(get_X_path("test", group), X_test_dict[group])

    # Save Targets and IDs
    np.save(get_y_path("train"), y_train)
    np.save(get_y_path("val"), y_val)
    np.save(get_id_path("test"), ids_test)

    return X_train_dict, y_train, X_val_dict, y_val, X_test_dict, ids_test
