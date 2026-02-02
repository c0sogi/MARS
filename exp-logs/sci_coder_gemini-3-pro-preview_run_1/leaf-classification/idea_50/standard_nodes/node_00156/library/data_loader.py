import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import METADATA_DIR, ID_COL, TARGET_COL, IMAGE_PATH_COL
from library.feature_engineering import load_and_process_data


def load_data(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Performs 'Subtractive Fusion' via the feature_engineering module.
    Applies the Inductive Preprocessing Pipeline:
        1. Yeo-Johnson Power Transformation (standardize=False)
        2. Standard Scaling
    Fitted ONLY on the training set to prevent leakage.

    Args:
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
            X_*: pd.DataFrame containing processed, transformed features (float64).
            y_*: pd.Series containing target labels.
            test_ids: pd.Series containing test image IDs.
    """

    # 1. Define Paths
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    # 2. Load and Process Data (Feature Extraction & Subtractive Fusion)
    # This step handles image processing, caching, dropping shape cols, and adding geometric scalars.
    print("Loading and processing Training set...")
    df_train = load_and_process_data(train_path, load_cached_data=load_cached_data)

    print("Loading and processing Validation set...")
    df_val = load_and_process_data(val_path, load_cached_data=load_cached_data)

    print("Loading and processing Test set...")
    df_test = load_and_process_data(test_path, load_cached_data=load_cached_data)

    # 3. Separate Features and Targets
    # Identify non-feature columns to exclude
    exclude_cols = {ID_COL, TARGET_COL, IMAGE_PATH_COL}

    # Determine feature columns from training set
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Enforce Alphanumeric Column Ordering (Deterministic Schema)
    feature_cols = sorted(feature_cols)

    # Prepare Train
    X_train = df_train[feature_cols].copy()
    y_train = df_train[TARGET_COL].copy()

    # Prepare Val
    X_val = df_val[feature_cols].copy()
    y_val = df_val[TARGET_COL].copy()

    # Prepare Test
    X_test = df_test[feature_cols].copy()
    test_ids = df_test[ID_COL].copy()

    # Ensure all X data is float64 before transformation
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # 4. Inductive Preprocessing Pipeline
    print("Applying Inductive Preprocessing Pipeline (Yeo-Johnson + StandardScaler)...")

    # Step A: Yeo-Johnson Power Transformation (standardize=False)
    # Fit on Train ONLY
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train_pt = pt.fit_transform(X_train)
    X_val_pt = pt.transform(X_val)
    X_test_pt = pt.transform(X_test)

    # Step B: Standard Scaling
    # Fit on Train (transformed) ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pt)
    X_val_scaled = scaler.transform(X_val_pt)
    X_test_scaled = scaler.transform(X_test_pt)

    # Convert back to DataFrames to preserve column names and structure
    X_train_final = pd.DataFrame(
        X_train_scaled, columns=feature_cols, index=X_train.index
    )
    X_val_final = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    X_test_final = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    print(f"Data Loading Complete.")
    print(f"Train Shape: {X_train_final.shape}")
    print(f"Val Shape:   {X_val_final.shape}")
    print(f"Test Shape:  {X_test_final.shape}")

    return X_train_final, y_train, X_val_final, y_val, X_test_final, test_ids
