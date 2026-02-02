import os
import json
import numpy as np
import pandas as pd
from library.config import CACHE_DIR
from library.data_handler import get_train_data, get_val_data, get_test_data


class TargetTransformer:
    """
    Applies log(1+y) transformation to targets and inverts it.
    This helps in stabilizing variance for regression targets that are strictly positive.
    """

    def __init__(self):
        pass

    def transform(self, y):
        """
        Apply log1p transformation: z = log(1 + y)
        """
        return np.log1p(y)

    def inverse_transform(self, z):
        """
        Apply expm1 transformation: y = exp(z) - 1
        """
        return np.expm1(z)


class FeatureCleaner:
    """
    Drops constant or quasi-constant columns and ensures numeric consistency.
    """

    def __init__(self):
        self.cols_to_drop = []
        self.fitted = False

    def fit(self, df):
        """
        Identifies columns to drop based on the provided DataFrame (usually training set).
        """
        self.cols_to_drop = []
        # Identify constant columns (1 unique value, ignoring NaNs)
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                self.cols_to_drop.append(col)

        self.fitted = True
        return self

    def transform(self, df):
        """
        Drops identified columns and ensures data is numeric/NaN.
        """
        if not self.fitted:
            raise RuntimeError("FeatureCleaner must be fitted before transform")

        # Drop columns
        df_clean = df.drop(columns=self.cols_to_drop, errors="ignore")

        # Replace infinite values with NaN to ensure XGBoost compatibility
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        return df_clean


def load_and_preprocess_data(load_cached_data=True, sample_size=None):
    """
    Loads features via data_handler, cleans them, and prepares X/y matrices.
    Caches the cleaned dataframes to CACHE_DIR.

    Args:
        load_cached_data (bool): If True, try to load from parquet cache.
        sample_size (int): Optional sample size for debugging.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, test_ids)
        where X are cleaned DataFrames and y are DataFrames with targets.
    """
    # Define cache filenames
    # We include sample_size in filename if it's set, to avoid loading partial data as full data
    suffix = "" if sample_size is None else f"_sample_{sample_size}"
    cache_train_path = os.path.join(CACHE_DIR, f"train_cleaned{suffix}.parquet")
    cache_val_path = os.path.join(CACHE_DIR, f"val_cleaned{suffix}.parquet")
    cache_test_path = os.path.join(CACHE_DIR, f"test_cleaned{suffix}.parquet")
    cache_state_path = os.path.join(CACHE_DIR, f"preprocessor_state{suffix}.json")

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    ):

        print(f"Loading preprocessed data from cache: {cache_train_path}...")
        train_df = pd.read_parquet(cache_train_path)
        val_df = pd.read_parquet(cache_val_path)
        test_df = pd.read_parquet(cache_test_path)

        target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

        # Separate features and targets
        y_train = train_df[target_cols]
        X_train = train_df.drop(columns=target_cols)

        y_val = val_df[target_cols]
        X_val = val_df.drop(columns=target_cols)

        test_ids = test_df["id"]
        X_test = test_df.drop(columns=["id"])

        return (X_train, y_train), (X_val, y_val), (X_test, test_ids)

    # 2. Compute from Scratch
    print("Preprocessing data from scratch...")

    # Load raw data (features + metadata)
    # data_handler handles the feature generation caching
    df_train_raw = get_train_data(
        load_cached_data=load_cached_data, sample_size=sample_size
    )
    df_val_raw = get_val_data(
        load_cached_data=load_cached_data, sample_size=sample_size
    )
    df_test_raw = get_test_data(
        load_cached_data=load_cached_data, sample_size=sample_size
    )

    # Define columns to exclude from X
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    meta_cols_to_drop = ["id", "file_path"]

    # Prepare Train
    y_train = df_train_raw[target_cols]
    X_train_raw = df_train_raw.drop(
        columns=target_cols + meta_cols_to_drop, errors="ignore"
    )

    # Prepare Val
    y_val = df_val_raw[target_cols]
    X_val_raw = df_val_raw.drop(
        columns=target_cols + meta_cols_to_drop, errors="ignore"
    )

    # Prepare Test
    test_ids = df_test_raw["id"]
    X_test_raw = df_test_raw.drop(columns=meta_cols_to_drop, errors="ignore")

    # Fit Cleaner on Train
    cleaner = FeatureCleaner()
    cleaner.fit(X_train_raw)

    # Transform all sets
    X_train_clean = cleaner.transform(X_train_raw)
    X_val_clean = cleaner.transform(X_val_raw)
    X_test_clean = cleaner.transform(X_test_raw)

    # 3. Save to Cache
    # Recombine for saving to keep alignment simple in parquet
    train_save = X_train_clean.copy()
    for col in target_cols:
        train_save[col] = y_train[col].values

    val_save = X_val_clean.copy()
    for col in target_cols:
        val_save[col] = y_val[col].values

    test_save = X_test_clean.copy()
    test_save["id"] = test_ids.values

    train_save.to_parquet(cache_train_path)
    val_save.to_parquet(cache_val_path)
    test_save.to_parquet(cache_test_path)

    # Save cleaner state
    with open(cache_state_path, "w") as f:
        json.dump({"cols_dropped": cleaner.cols_to_drop}, f)

    print(f"Preprocessed data saved to {CACHE_DIR}")

    return (X_train_clean, y_train), (X_val_clean, y_val), (X_test_clean, test_ids)
