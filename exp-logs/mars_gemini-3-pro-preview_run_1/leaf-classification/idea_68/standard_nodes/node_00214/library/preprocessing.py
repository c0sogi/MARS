import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
import library.config as config
import library.utils as utils

# Ensure deterministic behavior
utils.set_seed(config.SEED)


class SanitizedTransformer:
    """
    Implements the Inductive Preprocessing Pipeline with a Sanitization Barrier.

    Pipeline Steps:
    1. Sanitization: VarianceThreshold(threshold=0) to remove constant features.
    2. Transformation: Yeo-Johnson Power Transformation (standardize=False).
    3. Scaling: StandardScaler.

    All operations are performed in float64 precision.
    """

    def __init__(self):
        self.variance_selector = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
        self.power_transformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        self.scaler = StandardScaler()
        self.feature_mask = None

    def fit(self, X):
        """
        Fits the pipeline on the provided data (Training set).
        """
        # Ensure input is float64
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # 1. Sanitization: Fit VarianceThreshold
        self.variance_selector.fit(X)
        self.feature_mask = self.variance_selector.get_support()

        # Transform X to remove constant features for the next steps
        X_sanitized = self.variance_selector.transform(X)

        # 2. Transformation: Fit PowerTransformer
        self.power_transformer.fit(X_sanitized)
        X_transformed = self.power_transformer.transform(X_sanitized)

        # 3. Scaling: Fit StandardScaler
        self.scaler.fit(X_transformed)

        return self

    def transform(self, X):
        """
        Applies the fitted pipeline to new data.
        """
        # Ensure input is float64
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # 1. Sanitization
        if self.feature_mask is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")

        # Use the stored mask to filter features
        X_sanitized = self.variance_selector.transform(X)

        # 2. Transformation
        X_transformed = self.power_transformer.transform(X_sanitized)

        # 3. Scaling
        X_scaled = self.scaler.transform(X_transformed)

        return X_scaled.astype(config.FLOAT_PRECISION)


def get_preprocessed_data(train_df, val_df, test_df, load_cached_data=True):
    """
    Prepares the feature matrices and target vectors for the model.

    Logic:
    1. Checks for cached .npy files.
    2. If not found or reload forced:
       - Extracts features and targets.
       - Encodes targets.
       - Fits SanitizedTransformer on Train.
       - Transforms Train, Val, Test.
       - Caches results to disk.

    Args:
        train_df, val_df, test_df: DataFrames containing fused features.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, classes)
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "X_train_sanitized.npy"),
        "y_train": os.path.join(cache_dir, "y_train_encoded.npy"),
        "X_val": os.path.join(cache_dir, "X_val_sanitized.npy"),
        "y_val": os.path.join(cache_dir, "y_val_encoded.npy"),
        "X_test": os.path.join(cache_dir, "X_test_sanitized.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # 1. Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            print("Loading preprocessed data from cache...")
            try:
                X_train = np.load(paths["X_train"])
                y_train = np.load(paths["y_train"])
                X_val = np.load(paths["X_val"])
                y_val = np.load(paths["y_val"])
                X_test = np.load(paths["X_test"])
                classes = np.load(paths["classes"], allow_pickle=True)

                # Validation Check: Cite debug_lesson_2 (Invalidate Stale Cache)
                if (
                    (len(X_train) != len(train_df))
                    or (len(X_val) != len(val_df))
                    or (len(X_test) != len(test_df))
                ):
                    print(
                        f"Cache mismatch detected (Train: {len(X_train)} vs {len(train_df)}). Invalidating cache..."
                    )
                else:
                    return X_train, y_train, X_val, y_val, X_test, classes
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cache missing or incomplete. Processing from scratch...")

    # 2. Extract Raw Features and Targets
    # Ensure strict column ordering based on config
    feature_cols = config.ALL_FEATURES

    print(f"Extracting {len(feature_cols)} features...")
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[config.TARGET_COL].values

    X_val_raw = val_df[feature_cols].values
    y_val_raw = val_df[config.TARGET_COL].values

    X_test_raw = test_df[feature_cols].values
    # Test set does not have targets

    # 3. Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    # Filter validation set to ensure only classes seen in training are present
    # This prevents errors when using small debug subsets (Cite debug_lesson_1)
    val_mask = np.isin(y_val_raw, le.classes_)
    if not np.all(val_mask):
        print(f"Filtering {np.sum(~val_mask)} validation samples with unseen labels.")
        X_val_raw = X_val_raw[val_mask]
        y_val_raw = y_val_raw[val_mask]

    y_val = le.transform(y_val_raw)
    classes = le.classes_

    # 4. Apply Inductive Preprocessing Pipeline
    print("Fitting SanitizedTransformer on Training Data...")
    transformer = SanitizedTransformer()
    transformer.fit(X_train_raw)

    print("Transforming datasets...")
    X_train = transformer.transform(X_train_raw)
    X_val = transformer.transform(X_val_raw)
    X_test = transformer.transform(X_test_raw)

    # 5. Save to Cache
    print(f"Saving preprocessed data to {cache_dir}...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, classes
