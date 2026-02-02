import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    FEATURE_PREFIXES,
    NUMERIC_DTYPE,
    TARGET_COL,
    ID_COL,
)


class SemanticPreprocessor:
    """
    Handles the semantic splitting and high-precision preprocessing of leaf features.
    Applies Yeo-Johnson transformation followed by Standard Scaling per feature group.
    """

    def __init__(self):
        self.pt_transformers = {}
        self.ss_transformers = {}
        for group in FEATURE_PREFIXES:
            # Yeo-Johnson handles skewness; standardization is handled explicitly afterwards
            self.pt_transformers[group] = PowerTransformer(
                method="yeo-johnson", standardize=False
            )
            self.ss_transformers[group] = StandardScaler()

    def _get_group_columns(self, df, group):
        """
        Extracts and sorts column names for a specific feature group.
        Assumes columns are named like 'margin_1', 'margin_2', etc.
        """
        cols = [c for c in df.columns if c.startswith(f"{group}_")]
        # Sort numerically by the suffix (e.g., margin_2 before margin_10)
        cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
        return cols

    def fit(self, df):
        """
        Fits the transformers on the training data.
        """
        for group in FEATURE_PREFIXES:
            cols = self._get_group_columns(df, group)
            # Ensure strict float64 precision
            X = df[cols].values.astype(NUMERIC_DTYPE)

            # Fit PowerTransformer
            self.pt_transformers[group].fit(X)
            # Transform to get intermediate state for StandardScaler fitting
            X_pt = self.pt_transformers[group].transform(X)
            # Fit StandardScaler
            self.ss_transformers[group].fit(X_pt)
        return self

    def transform(self, df):
        """
        Transforms data using the fitted transformers.
        Returns a dictionary of arrays: {'margin': X_m, 'shape': X_s, 'texture': X_t}
        """
        output = {}
        for group in FEATURE_PREFIXES:
            cols = self._get_group_columns(df, group)
            X = df[cols].values.astype(NUMERIC_DTYPE)

            # Apply transformations
            X_pt = self.pt_transformers[group].transform(X)
            X_ss = self.ss_transformers[group].transform(X_pt)

            output[group] = X_ss
        return output


def load_and_process_data(load_cached_data=True):
    """
    Loads data, performs semantic splitting and preprocessing, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.

    Returns:
        tuple: (data_dict, classes)
            data_dict: Nested dictionary containing 'train', 'val', 'test' splits.
                       Each split has 'X' (dict of group arrays), 'y' (labels), and 'ids'.
            classes: Array of class names corresponding to label indices.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    splits = ["train", "val", "test"]

    # Check if cache exists
    cache_valid = True
    expected_files = []

    # Define expected files for validation
    for split in splits:
        for group in FEATURE_PREFIXES:
            expected_files.append(f"X_{split}_{group}.npy")
        expected_files.append(f"ids_{split}.npy")
        if split != "test":
            expected_files.append(f"y_{split}.npy")
    expected_files.append("classes.npy")

    for fname in expected_files:
        if not os.path.exists(os.path.join(WORKING_DIR, fname)):
            cache_valid = False
            break

    if load_cached_data and cache_valid:
        print("Loading pre-processed data from cache...")
        data = {}
        for split in splits:
            data[split] = {}
            # Load Features
            X_dict = {}
            for group in FEATURE_PREFIXES:
                X_dict[group] = np.load(
                    os.path.join(WORKING_DIR, f"X_{split}_{group}.npy")
                )
            data[split]["X"] = X_dict

            # Load IDs
            data[split]["ids"] = np.load(os.path.join(WORKING_DIR, f"ids_{split}.npy"))

            # Load Targets
            if split != "test":
                data[split]["y"] = np.load(os.path.join(WORKING_DIR, f"y_{split}.npy"))

        classes = np.load(os.path.join(WORKING_DIR, "classes.npy"), allow_pickle=True)
        return data, classes

    print("Cache missing or reload requested. Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[TARGET_COL])
    y_val = le.transform(df_val[TARGET_COL])
    classes = le.classes_

    # Extract IDs
    ids_train = df_train[ID_COL].values
    ids_val = df_val[ID_COL].values
    ids_test = df_test[ID_COL].values

    # Preprocessing
    # Fit only on TRAIN
    preprocessor = SemanticPreprocessor()
    preprocessor.fit(df_train)

    # Transform all splits
    X_train_dict = preprocessor.transform(df_train)
    X_val_dict = preprocessor.transform(df_val)
    X_test_dict = preprocessor.transform(df_test)

    # Save to Cache
    # Helper to save dict of arrays
    def save_split_features(split_name, x_dict):
        for group, arr in x_dict.items():
            np.save(os.path.join(WORKING_DIR, f"X_{split_name}_{group}.npy"), arr)

    # Save Train
    save_split_features("train", X_train_dict)
    np.save(os.path.join(WORKING_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(WORKING_DIR, "ids_train.npy"), ids_train)

    # Save Val
    save_split_features("val", X_val_dict)
    np.save(os.path.join(WORKING_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(WORKING_DIR, "ids_val.npy"), ids_val)

    # Save Test
    save_split_features("test", X_test_dict)
    np.save(os.path.join(WORKING_DIR, "ids_test.npy"), ids_test)

    # Save Classes
    np.save(os.path.join(WORKING_DIR, "classes.npy"), classes)

    # Construct Return Dictionary
    data = {
        "train": {"X": X_train_dict, "y": y_train, "ids": ids_train},
        "val": {"X": X_val_dict, "y": y_val, "ids": ids_val},
        "test": {"X": X_test_dict, "ids": ids_test},
    }

    return data, classes
