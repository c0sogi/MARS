import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config, data_loader


class GlobalPreprocessor:
    """
    Manages a global PowerTransformer and StandardScaler for the entire feature set.
    Treats features as a monolithic vector to preserve cross-correlations.
    Cite Lesson 82: Treat features as a flat vector when using global covariance models.
    """

    def __init__(self):
        self.pt = None
        self.ss = None
        self.feature_cols = None

    def fit(self, df_train):
        """
        Fits transformers on the training data.
        """
        # Identify feature columns
        # Cite Lesson 29: Enforce Explicit Feature Ordering for Numerical Stability
        exclude = {config.ID_COLUMN, config.TARGET_COLUMN, "file_path", "full_path"}
        self.feature_cols = sorted([c for c in df_train.columns if c not in exclude])

        # Extract data and ensure float64 precision
        # Cite Lesson 73: Always maintain float64 throughout the pipeline
        X = df_train[self.feature_cols].values.astype(config.FLOAT_PRECISION)

        # 1. Yeo-Johnson Power Transformation (standardize=False)
        # Cite Lesson 25: Avoid redundant normalization (standardize=False)
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
        if self.feature_cols is None:
            raise ValueError("Preprocessor has not been fitted.")

        X = df[self.feature_cols].values.astype(config.FLOAT_PRECISION)

        # Apply transformations
        X_pt = self.pt.transform(X)
        X_ss = self.ss.transform(X_pt)

        # Cite Lesson 46: Defer precision reduction (though we keep float64 here per Lesson 73)
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
            print(
                f"Debug size mismatch (Cache: {len(data_out['y_train'])}, Expected: {expected_size}). Recomputing..."
            )
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

    print(f"Saving preprocessed data to {config.CACHE_DIR}...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    np.save(get_X_path("train"), X_train)
    np.save(get_X_path("val"), X_val)
    np.save(get_X_path("test"), X_test)
    np.save(get_y_path("train"), y_train)
    np.save(get_y_path("val"), y_val)
    np.save(get_id_path("test"), ids_test)

    return X_train, y_train, X_val, y_val, X_test, ids_test
