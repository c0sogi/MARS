import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from library.utils import set_seed
from library.image_processing import process_dataset

# Constants
CACHE_DIR = "./working/idea_optimized"
METADATA_DIR = "./metadata"


def _load_and_merge(metadata_filename, load_cached_data=True):
    """
    Helper function to load metadata, extract geometric features, and merge them.
    Returns a DataFrame with features and targets (if available), and a list of feature columns.
    """
    metadata_path = os.path.join(METADATA_DIR, metadata_filename)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 1. Load Metadata (Tabular Features + Labels)
    df_meta = pd.read_csv(metadata_path)

    # 2. Load/Compute Geometric Features
    df_geo = process_dataset(metadata_path, load_cached_data=load_cached_data)

    # 3. Merge on ID
    # Verify IDs match before merging to be safe
    # (Inner join on 'id' handles alignment)
    df_merged = pd.merge(df_meta, df_geo, on="id", how="inner")

    # 4. Identify Feature Columns
    # Exclude non-feature columns
    exclude_cols = ["id", "species", "file_path"]
    feature_cols = [c for c in df_merged.columns if c not in exclude_cols]

    # 5. Enforce Alphanumeric Column Ordering (Deterministic Schema)
    feature_cols.sort()

    return df_merged, feature_cols


def load_dataset(load_cached_data=True, max_samples=None):
    """
    Loads, merges, sanitizes, and transforms the dataset.

    Pipeline:
    1. Merge Tabular + Geometric features.
    2. Split into Train/Val/Test.
    3. VarianceThreshold (threshold=0) [Fit Train].
    4. PowerTransformer (yeo-johnson, standardize=False) [Fit Train].
    5. StandardScaler [Fit Train].
    6. Cache results (if max_samples is None).

    Args:
        load_cached_data (bool): Whether to load pre-processed arrays from disk.
        max_samples (int, optional): Number of samples to use for debugging.
                                     If set, caching is disabled.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache filenames
    files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # 1. Try Loading from Cache
    # Only load cache if max_samples is None (full dataset) and requested
    if load_cached_data and max_samples is None:
        if all(os.path.exists(f) for f in files.values()):
            print("Loading processed dataset from cache...")
            X_train = np.load(files["X_train"])
            y_train = np.load(files["y_train"])
            X_val = np.load(files["X_val"])
            y_val = np.load(files["y_val"])
            X_test = np.load(files["X_test"])
            test_ids = np.load(files["test_ids"])
            classes = np.load(files["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Processing dataset from scratch...")

    # 2. Load Raw Data (Train, Val, Test)
    # Pass load_cached_data to _load_and_merge to utilize image feature caching
    df_train, feat_cols = _load_and_merge(
        "train.csv", load_cached_data=load_cached_data
    )
    df_val, _ = _load_and_merge("val.csv", load_cached_data=load_cached_data)
    df_test, _ = _load_and_merge("test.csv", load_cached_data=load_cached_data)

    # 3. Apply max_samples (Debugging)
    if max_samples is not None:
        print(f"Subsampling dataset to {max_samples} samples...")
        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows
        # 1. Select a pool of classes large enough to fill max_samples
        # We use top 50 classes to ensure sufficient data density and overlap
        pool_classes = df_train["species"].value_counts().index[:50]
        df_train = df_train[df_train["species"].isin(pool_classes)]

        # 2. Slice train to exact limit
        df_train = df_train.iloc[:max_samples]

        # 3. Restrict val to the classes actually present in the sliced train set
        actual_classes = df_train["species"].unique()
        df_val = df_val[df_val["species"].isin(actual_classes)].iloc[:max_samples]

        # 4. Slice test (no labels)
        df_test = df_test.iloc[:max_samples]

    # 4. Extract Targets and IDs
    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    # Extract IDs
    test_ids = df_test["id"].values.astype(np.int64)

    # 5. Extract Features (Ensure float64)
    X_train = df_train[feat_cols].values.astype(np.float64)
    X_val = df_val[feat_cols].values.astype(np.float64)
    X_test = df_test[feat_cols].values.astype(np.float64)

    # 6. Sanitization & Transformation Pipeline
    # Inductive Fit: Fit ONLY on Train, Transform All

    # A. Variance Threshold (Sanitization Barrier)
    # Removes constant features that would break scaling
    vt = VarianceThreshold(threshold=0)
    X_train = vt.fit_transform(X_train)
    X_val = vt.transform(X_val)
    X_test = vt.transform(X_test)

    # B. Power Transformation (Yeo-Johnson)
    # Stabilizes variance and minimizes skewness
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train = pt.fit_transform(X_train)
    X_val = pt.transform(X_val)
    X_test = pt.transform(X_test)

    # C. Standard Scaling
    # Centers and scales to unit variance
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)
    X_test = ss.transform(X_test)

    # 7. Save to Cache
    # Only save if we processed the full dataset
    if max_samples is None:
        print("Saving processed dataset to cache...")
        np.save(files["X_train"], X_train)
        np.save(files["y_train"], y_train)
        np.save(files["X_val"], X_val)
        np.save(files["y_val"], y_val)
        np.save(files["X_test"], X_test)
        np.save(files["test_ids"], test_ids)
        np.save(files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
