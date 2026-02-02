import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from library import image_utils

# Constants
CACHE_DIR = "./working/idea_41"
METADATA_DIR = "./metadata"


def load_and_merge_data(load_cached_data=True):
    """
    Loads metadata, extracts provided features, computes/loads morphometric features,
    and merges them into combined feature matrices.

    Args:
        load_cached_data (bool): If True, attempts to load from local cache first.

    Returns:
        X_train (np.ndarray): Combined features for training (float64).
        y_train (np.ndarray): Species labels for training.
        X_val (np.ndarray): Combined features for validation (float64).
        y_val (np.ndarray): Species labels for validation.
        X_test (np.ndarray): Combined features for testing (float64).
        test_ids (np.ndarray): IDs for the test set.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "merged_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading merged data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["test_ids"],
        )

    print("Generating merged data from scratch...")

    # 1. Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 2. Identify Provided Feature Columns
    # Filter out non-feature columns.
    # Provided features are usually margin_*, shape_*, texture_*
    exclude_cols = ["id", "species", "image_path"]
    # We preserve the order found in the CSV
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # 3. Get Morphometric Features
    # This function handles its own caching for the raw extraction part
    train_morph, val_morph, test_morph = image_utils.get_morphometric_features(
        metadata_dir=METADATA_DIR, load_cached_data=load_cached_data
    )

    # 4. Merge Data
    def process_split(df_meta, df_morph, is_test=False):
        # Merge on 'id'
        merged = pd.merge(df_meta, df_morph, on="id", how="left")

        # Extract Provided Features (Global View)
        X_provided = merged[feature_cols].values.astype(np.float64)

        # Extract Morphometric Features
        # Morph columns are those in df_morph that are not 'id'
        morph_cols = [c for c in df_morph.columns if c != "id"]
        # Sort to ensure deterministic column order
        morph_cols.sort()
        X_morph = merged[morph_cols].values.astype(np.float64)

        # Combine (Provided + Morphometric)
        X_combined = np.hstack([X_provided, X_morph])

        if is_test:
            return X_combined, merged["id"].values
        else:
            return X_combined, merged["species"].values

    X_train, y_train = process_split(train_df, train_morph)
    X_val, y_val = process_split(val_df, val_morph)
    X_test, test_ids = process_split(test_df, test_morph, is_test=True)

    # 5. Save to Cache
    print(f"Saving merged data to {cache_path}")
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        test_ids=test_ids,
    )

    return X_train, y_train, X_val, y_val, X_test, test_ids


def apply_topology(X_train, X_val, X_test, topology_type="marginal", random_state=42):
    """
    Applies the specified Gaussianization topology to the feature matrices.

    Args:
        X_train, X_val, X_test (np.ndarray): Input feature matrices.
        topology_type (str): 'marginal' or 'iterative'.
        random_state (int): Seed for reproducibility (used in PCA).

    Returns:
        X_train_trans, X_val_trans, X_test_trans (np.ndarray): Transformed matrices (float64).
    """
    print(f"Applying topology: {topology_type}")

    # Ensure float64 precision for numerical stability
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    if topology_type == "marginal":
        # Topology A: Marginal Gaussian Anchors
        # Standard PowerTransformer (Yeo-Johnson)
        pt = PowerTransformer(method="yeo-johnson", standardize=True)

        X_train_trans = pt.fit_transform(X_train)
        X_val_trans = pt.transform(X_val)
        X_test_trans = pt.transform(X_test)

        return X_train_trans, X_val_trans, X_test_trans

    elif topology_type == "iterative":
        # Topology B: Iterative Gaussian Experts
        # Pipeline: PowerTransformer -> PCA -> PowerTransformer

        # Stage 1: Stabilize marginal variances
        pt1 = PowerTransformer(method="yeo-johnson", standardize=True)
        X_train_s1 = pt1.fit_transform(X_train)
        X_val_s1 = pt1.transform(X_val)
        X_test_s1 = pt1.transform(X_test)

        # Stage 2: Rotate using PCA (no whitening)
        # This aligns the data with principal axes, exposing non-Gaussian structures
        # to the subsequent marginal transform.
        pca = PCA(n_components=None, whiten=False, random_state=random_state)
        X_train_s2 = pca.fit_transform(X_train_s1)
        X_val_s2 = pca.transform(X_val_s1)
        X_test_s2 = pca.transform(X_test_s1)

        # Stage 3: Apply second Power Transform to the rotated components
        pt2 = PowerTransformer(method="yeo-johnson", standardize=True)
        X_train_s3 = pt2.fit_transform(X_train_s2)
        X_val_s3 = pt2.transform(X_val_s2)
        X_test_s3 = pt2.transform(X_test_s2)

        return X_train_s3, X_val_s3, X_test_s3

    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
