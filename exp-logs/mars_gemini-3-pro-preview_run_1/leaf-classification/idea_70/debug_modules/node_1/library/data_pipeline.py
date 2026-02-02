import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    SEED,
    set_seed,
    NP_FLOAT_PRECISION,
)
from library.feature_extraction import get_geometric_features

# Ensure reproducibility
set_seed(SEED)


class PipelineProcessor:
    """
    Encapsulates the preprocessing pipeline:
    1. VarianceThreshold(0) to remove constant features.
    2. PowerTransformer(yeo-johnson, standardize=False) to stabilize variance.
    3. StandardScaler to normalize to N(0, 1).

    Ensures all operations use high precision (float64).
    """

    def __init__(self):
        self.vt = VarianceThreshold(threshold=0.0)
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.ss = StandardScaler()

    def fit(self, X):
        """
        Fits the pipeline on the training data.
        X: numpy array of shape (n_samples, n_features)
        """
        # 1. Variance Threshold
        # Removes features with zero variance
        X_vt = self.vt.fit_transform(X)

        # 2. Power Transformer
        # Stabilizes variance and minimizes skewness
        self.pt.fit(X_vt)

        # 3. Standard Scaler
        # We must transform X via VT and PT before fitting SS to get correct mean/std
        X_pt = self.pt.transform(X_vt)
        self.ss.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.
        """
        X_vt = self.vt.transform(X)
        X_pt = self.pt.transform(X_vt)
        X_ss = self.ss.transform(X_pt)
        return X_ss.astype(NP_FLOAT_PRECISION)


def load_and_fuse_data(load_cached_data=True):
    """
    Loads metadata, extracts/loads geometric features, and fuses them.
    Returns fused DataFrames for train, val, and test.
    """
    cache_train = os.path.join(CACHE_DIR, "fused_train.parquet")
    cache_val = os.path.join(CACHE_DIR, "fused_val.parquet")
    cache_test = os.path.join(CACHE_DIR, "fused_test.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading fused datasets from cache...")
        return (
            pd.read_parquet(cache_train),
            pd.read_parquet(cache_val),
            pd.read_parquet(cache_test),
        )

    print("Generating fused datasets (merging metadata and geometric features)...")

    # Load Metadata
    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    df_val_meta = pd.read_csv(VAL_META_PATH)
    df_test_meta = pd.read_csv(TEST_META_PATH)

    # Load Geometric Features (handles its own caching)
    geo_train, geo_val, geo_test = get_geometric_features(
        load_cached_data=load_cached_data
    )

    def fuse(df_meta, df_geo):
        # Ensure ID types match for merging
        df_meta = df_meta.copy()
        df_geo = df_geo.copy()
        df_meta["id"] = df_meta["id"].astype(int)
        df_geo["id"] = df_geo["id"].astype(int)

        # Merge
        fused = pd.merge(df_meta, df_geo, on="id", how="left")

        # Fill missing geometric features with 0 (safety net)
        geo_cols = [c for c in df_geo.columns if c != "id"]
        fused[geo_cols] = fused[geo_cols].fillna(0.0)

        return fused

    df_train_fused = fuse(df_train_meta, geo_train)
    df_val_fused = fuse(df_val_meta, geo_val)
    df_test_fused = fuse(df_test_meta, geo_test)

    # Save to cache
    print("Saving fused datasets to cache...")
    df_train_fused.to_parquet(cache_train, index=False)
    df_val_fused.to_parquet(cache_val, index=False)
    df_test_fused.to_parquet(cache_test, index=False)

    return df_train_fused, df_val_fused, df_test_fused


def get_data_pipeline(load_cached_data=True, debug_subset_size=None):
    """
    Main entry point for the data pipeline.

    Args:
        load_cached_data (bool): Whether to use cached intermediate files.
        debug_subset_size (int, optional): If set, limits the dataset size for debugging.

    Returns:
        dict: Contains processed X/y arrays, the label encoder, and feature names.
    """
    # 1. Load and Fuse
    df_train, df_val, df_test = load_and_fuse_data(load_cached_data)

    # 2. Debug Subsetting
    if debug_subset_size is not None:
        print(f"DEBUG: Subsetting data to {debug_subset_size} samples.")
        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows
        # Naive slicing creates disjoint label sets in high-cardinality data.
        # We must ensure validation set only contains classes present in the training subset.
        df_train = df_train.head(debug_subset_size)

        valid_species = df_train["species"].unique()
        df_val = df_val[df_val["species"].isin(valid_species)].head(debug_subset_size)

        df_test = df_test.head(debug_subset_size)

    # 3. Identify Feature Columns
    # We want columns starting with margin, shape, texture, or geo
    all_cols = df_train.columns
    feature_cols = [
        c
        for c in all_cols
        if c.startswith("margin")
        or c.startswith("shape")
        or c.startswith("texture")
        or c.startswith("geo")
    ]
    feature_cols.sort()  # Deterministic order

    print(f"Total features selected: {len(feature_cols)}")

    # 4. Extract Raw Arrays (float64)
    X_train_raw = df_train[feature_cols].values.astype(NP_FLOAT_PRECISION)
    y_train_raw = df_train["species"].values

    X_val_raw = df_val[feature_cols].values.astype(NP_FLOAT_PRECISION)
    y_val_raw = df_val["species"].values

    X_test_raw = df_test[feature_cols].values.astype(NP_FLOAT_PRECISION)
    test_ids = df_test["id"].values

    # 5. Encode Targets
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)

    # Handle unseen labels in validation if any (shouldn't happen with stratified split)
    y_val_enc = le.transform(y_val_raw)

    # 6. Apply Preprocessing Pipeline
    print("Fitting preprocessing pipeline on training data...")
    processor = PipelineProcessor()
    processor.fit(X_train_raw)

    print("Transforming datasets...")
    X_train_proc = processor.transform(X_train_raw)
    X_val_proc = processor.transform(X_val_raw)
    X_test_proc = processor.transform(X_test_raw)

    return {
        "train": (X_train_proc, y_train_enc),
        "val": (X_val_proc, y_val_enc),
        "test": (X_test_proc, test_ids),
        "label_encoder": le,
        "feature_names": feature_cols,
        "processor": processor,
    }
