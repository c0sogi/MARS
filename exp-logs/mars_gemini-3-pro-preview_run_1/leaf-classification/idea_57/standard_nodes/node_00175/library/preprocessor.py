import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import Config
from library.feature_extraction import process_dataset


class SanitizedTransformPipeline:
    """
    A preprocessing pipeline that implements the Sanitization Barrier,
    Inductive Transformation, and Scaling steps with strict float64 precision.
    """

    def __init__(self):
        # 1. Sanitization Barrier: Remove constant features
        self.sanitizer = VarianceThreshold(threshold=0.0)
        # 2. Inductive Transformation: Stabilize variance (Yeo-Johnson)
        self.transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        # 3. Scaling: Standardize to zero mean and unit variance
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fit the pipeline components sequentially.

        Args:
            X (array-like): Feature matrix.
            y (array-like, optional): Target labels.
        """
        # Ensure float64 precision
        X = X.astype(np.float64)

        # Fit Sanitizer
        X_sanitized = self.sanitizer.fit_transform(X)

        # Fit Transformer
        X_transformed = self.transformer.fit_transform(X_sanitized)

        # Fit Scaler
        self.scaler.fit(X_transformed)

        return self

    def transform(self, X):
        """
        Apply the pipeline transforms to new data.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Transformed feature matrix.
        """
        # Ensure float64 precision
        X = X.astype(np.float64)

        # Apply Sanitizer
        X_sanitized = self.sanitizer.transform(X)

        # Apply Transformer
        X_transformed = self.transformer.transform(X_sanitized)

        # Apply Scaler
        X_scaled = self.scaler.transform(X_transformed)

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one pass.
        """
        return self.fit(X, y).transform(X)


def load_data(subset, load_cached_data=True, max_samples=None):
    """
    Loads metadata, extracts geometric features, merges with tabular features,
    and returns the raw feature matrix X, labels y, and ids.

    Args:
        subset (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache.
        max_samples (int, optional): Maximum number of samples to load (for debugging).

    Returns:
        tuple: (X, y, ids, feature_names)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{subset}_combined.parquet")

    df = None

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            # If cache is corrupt, proceed to process from scratch
            df = None

    # 2. Process from Scratch if needed
    if df is None:
        if subset == "train":
            meta_path = Config.TRAIN_META
        elif subset == "val":
            meta_path = Config.VAL_META
        elif subset == "test":
            meta_path = Config.TEST_META
        else:
            raise ValueError(f"Unknown subset: {subset}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Extract Geometric Features
        # We process strictly from metadata and input dir, disabling internal cache
        # to ensure we manage the combined cache here.
        geo_df = process_dataset(
            metadata_df=df_meta,
            input_dir=Config.INPUT_DIR,
            cache_path=None,
            load_cached_data=False,
        )

        # Identify Tabular Features in Metadata
        # Columns starting with margin, shape, texture
        tab_cols = [
            c
            for c in df_meta.columns
            if any(c.startswith(prefix) for prefix in Config.TABULAR_FEATURES_PREFIX)
        ]

        # Construct final DataFrame
        # We start with ID and Species (if available)
        cols_to_keep = ["id"]
        if "species" in df_meta.columns:
            cols_to_keep.append("species")

        df_base = df_meta[cols_to_keep].reset_index(drop=True)
        df_tab = df_meta[tab_cols].reset_index(drop=True)

        # Handle geometric dataframe (remove ID if duplicated)
        if "id" in geo_df.columns:
            geo_df = geo_df.drop(columns=["id"])
        geo_df = geo_df.reset_index(drop=True)

        # Concatenate
        df_combined = pd.concat([df_base, df_tab, geo_df], axis=1)

        # Save to cache
        df_combined.to_parquet(cache_path, index=False)
        df = df_combined

    # Apply max_samples for debugging
    if max_samples is not None and max_samples < len(df):
        df = df.iloc[:max_samples]

    # 3. Prepare Outputs
    # Extract IDs
    ids = df["id"].values

    # Extract Labels if present
    y = None
    if "species" in df.columns:
        y = df["species"].values

    # Extract Features
    # Exclude non-feature columns
    exclude_cols = ["id", "species"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Enforce alphanumeric sort for deterministic order and memory layout
    feature_cols.sort()

    # Convert to float64 numpy array
    X = df[feature_cols].values.astype(np.float64)

    return X, y, ids, feature_cols
