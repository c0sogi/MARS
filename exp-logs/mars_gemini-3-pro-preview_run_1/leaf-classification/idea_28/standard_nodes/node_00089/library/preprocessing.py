import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config, data_loader


class GlobalPreprocessor:
    """
    Manages global PowerTransformer and StandardScaler instances for all features.
    Cite solution_lesson_node_00087: Prefer global shrinkage estimators over manual feature factorization.
    """

    def __init__(self):
        self.pt = None
        self.ss = None
        self.feature_cols = None

    def fit(self, df_train):
        """
        Fits transformers on the training data for all features globally.
        """
        # Identify feature columns
        exclude_cols = {
            config.ID_COLUMN,
            config.TARGET_COLUMN,
            "file_path",
            "full_path",
        }
        # Cite solution_lesson_node_00082: Treat features as a flat vector, alphanumeric sort
        self.feature_cols = sorted(
            [c for c in df_train.columns if c not in exclude_cols]
        )

        # Extract data and ensure float64 precision
        # Cite solution_lesson_node_00073: Maintain float64 precision
        X = df_train[self.feature_cols].values.astype(config.FLOAT_PRECISION)

        # 1. Yeo-Johnson Power Transformation (standardize=False)
        # Cite solution_lesson_node_00025: Disable internal standardization in PowerTransformer
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        X_pt = self.pt.fit_transform(X)

        # 2. Standard Scaling
        self.ss = StandardScaler()
        self.ss.fit(X_pt)

        return self

    def transform(self, df):
        """
        Applies the fitted transformers to the dataframe.
        """
        # Cite solution_lesson_node_00031: Explicit feature whitelisting
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        X = df[self.feature_cols].values.astype(config.FLOAT_PRECISION)

        # Apply transformations
        X_pt = self.pt.transform(X)
        X_ss = self.ss.transform(X_pt)

        # Return result ensuring precision
        return X_ss.astype(config.FLOAT_PRECISION)


def get_preprocessed_data(load_cached_data=True, debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Orchestrates loading, preprocessing, and caching of data.
    """
    # Define cache filenames helper
    splits = ["train", "val", "test"]

    def get_X_path(split):
        return os.path.join(config.CACHE_DIR, f"X_{split}.npy")

    def get_y_path(split):
        return os.path.join(config.CACHE_DIR, f"y_{split}.npy")

    def get_id_path(split):
        return os.path.join(config.CACHE_DIR, f"ids_{split}.npy")

    # Check cache integrity
    cache_complete = True
    for split in splits:
        if not os.path.exists(get_X_path(split)):
            cache_complete = False
            break
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

        for split in splits:
            data_out[f"X_{split}"] = np.load(get_X_path(split))

        data_out["y_train"] = np.load(get_y_path("train"), allow_pickle=True)
        data_out["y_val"] = np.load(get_y_path("val"), allow_pickle=True)
        data_out["ids_test"] = np.load(get_id_path("test"))

        # Validate cache consistency
        if debug_size is not None:
            expected_size = debug_size
        else:
            df_full_meta = pd.read_csv(
                config.TRAIN_DATA_PATH, usecols=[config.ID_COLUMN]
            )
            expected_size = len(df_full_meta)

        if len(data_out["y_train"]) != expected_size:
            print(f"Debug size mismatch. Recomputing...")
        else:
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

    df_train, df_val, df_test = data_loader.load_datasets(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    y_train = df_train[config.TARGET_COLUMN].values
    y_val = df_val[config.TARGET_COLUMN].values
    ids_test = df_test[config.ID_COLUMN].values

    preprocessor = GlobalPreprocessor()
    preprocessor.fit(df_train)

    X_train = preprocessor.transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    # Save to cache
    print(f"Saving preprocessed data to {config.CACHE_DIR}...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    np.save(get_X_path("train"), X_train)
    np.save(get_X_path("val"), X_val)
    np.save(get_X_path("test"), X_test)

    np.save(get_y_path("train"), y_train)
    np.save(get_y_path("val"), y_val)
    np.save(get_id_path("test"), ids_test)

    return X_train, y_train, X_val, y_val, X_test, ids_test
