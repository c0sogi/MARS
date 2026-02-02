import os
import numpy as np
import pandas as pd
from library.config import WORKING_DIR

# Constants
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
CACHE_DIR = WORKING_DIR  # Use the idea-specific working directory


def log_transform_targets(
    df: pd.DataFrame, targets: list = TARGET_COLS
) -> pd.DataFrame:
    """
    Applies log(1 + y) transformation to the target columns.
    """
    df_transformed = df.copy()
    for col in targets:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def inverse_log_transform(y_pred: np.ndarray) -> np.ndarray:
    """
    Applies exp(y) - 1 transformation to predictions to return to original scale.
    """
    return np.expm1(y_pred)


def get_numeric_features(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Identifies numeric feature columns, excluding specified target/id columns.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Select numeric types
    numeric_df = df.select_dtypes(include=[np.number])
    return [c for c in numeric_df.columns if c not in exclude_cols]


def remove_constant_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple:
    """
    Removes columns that have zero variance (constant values) in the training set.
    Applies the same removal to validation and test sets.
    """
    # Calculate standard deviation of training features
    std_series = X_train.std()

    # Identify constant columns (std == 0 or NaN)
    constant_cols = std_series[std_series.fillna(0) == 0].index.tolist()

    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant features based on training set.")
        X_train_clean = X_train.drop(columns=constant_cols)
        X_val_clean = X_val.drop(columns=constant_cols, errors="ignore")
        X_test_clean = X_test.drop(columns=constant_cols, errors="ignore")
        return X_train_clean, X_val_clean, X_test_clean

    return X_train, X_val, X_test


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    load_cached_data: bool = True,
) -> tuple:
    """
    Main preprocessing pipeline with caching.

    Steps:
    1. Check for cached preprocessed files.
    2. If not found or forced reload:
       a. Separate features (X) and targets (y).
       b. Log-transform targets.
       c. Drop non-numeric columns (if any) and targets from X.
       d. Remove constant features.
       e. Save to cache.
    3. Return X_train, y_train, X_val, y_val, X_test.
    """

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.parquet"),
        "y_train": os.path.join(CACHE_DIR, "y_train.parquet"),
        "X_val": os.path.join(CACHE_DIR, "X_val.parquet"),
        "y_val": os.path.join(CACHE_DIR, "y_val.parquet"),
        "X_test": os.path.join(CACHE_DIR, "X_test.parquet"),
    }

    # 1. Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading preprocessed data from {CACHE_DIR}...")
            X_train = pd.read_parquet(cache_files["X_train"])
            y_train = pd.read_parquet(cache_files["y_train"])
            X_val = pd.read_parquet(cache_files["X_val"])
            y_val = pd.read_parquet(cache_files["y_val"])
            X_test = pd.read_parquet(cache_files["X_test"])
            return X_train, y_train, X_val, y_val, X_test
        else:
            print("Cached preprocessed data not found. Processing from scratch...")
    else:
        print("Force reprocessing data...")

    # 2. Process Data

    # a. Log transform targets
    train_df_log = log_transform_targets(train_df, TARGET_COLS)
    val_df_log = log_transform_targets(val_df, TARGET_COLS)

    # b. Separate X and y
    # Identify feature columns (numeric, excluding targets)
    # Note: 'id' is the index, so it's handled automatically
    feature_cols = get_numeric_features(train_df, exclude_cols=TARGET_COLS)

    X_train = train_df_log[feature_cols].copy()
    y_train = train_df_log[TARGET_COLS].copy()

    X_val = val_df_log[feature_cols].copy()
    y_val = val_df_log[TARGET_COLS].copy()

    # Test set doesn't have targets
    # Ensure test set has same feature columns (fill missing with 0 if any, though unlikely with this pipeline)
    X_test = test_df.copy()
    # Filter to keep only numeric and relevant columns found in train
    valid_test_cols = [c for c in feature_cols if c in X_test.columns]
    X_test = X_test[valid_test_cols]

    # c. Remove constant features
    X_train, X_val, X_test = remove_constant_features(X_train, X_val, X_test)

    # Fill NaNs if any (simple imputation to be safe, though descriptors should be clean)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    # 3. Save to cache
    print(f"Saving preprocessed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    X_train.to_parquet(cache_files["X_train"])
    y_train.to_parquet(cache_files["y_train"])
    X_val.to_parquet(cache_files["X_val"])
    y_val.to_parquet(cache_files["y_val"])
    X_test.to_parquet(cache_files["X_test"])

    return X_train, y_train, X_val, y_val, X_test
