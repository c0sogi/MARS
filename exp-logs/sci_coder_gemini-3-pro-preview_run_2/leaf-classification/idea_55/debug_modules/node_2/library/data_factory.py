import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import METADATA_DIR, WORKING_DIR, FLOAT_PRECISION, RANDOM_SEED
from library.utils import set_seed
from library.image_processing import extract_morphometrics

# Define cache directory
CACHE_DIR = os.path.join(WORKING_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_paths(dataset_name):
    """Returns a dictionary of cache file paths for a given dataset."""
    return {
        "X": os.path.join(CACHE_DIR, f"X_{dataset_name}.npy"),
        "y": os.path.join(CACHE_DIR, f"y_{dataset_name}.npy"),
        "ids": os.path.join(CACHE_DIR, f"ids_{dataset_name}.npy"),
        "classes": os.path.join(CACHE_DIR, f"classes_{dataset_name}.npy"),
    }


def load_dataset(dataset_name, load_cached_data=True):
    """
    Loads and preprocesses the dataset (Train, Val, or Test).

    Combines Global Features (192 columns) and Morphometric Features (12 columns).
    Ensures strict float64 precision.

    Args:
        dataset_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        tuple: (X, y, ids, classes)
            X (np.ndarray): Feature matrix (N, 204).
            y (np.ndarray): Label array (N,) or None for test.
            ids (np.ndarray): ID array (N,).
            classes (np.ndarray): Array of class names (K,) or None.
    """
    set_seed(RANDOM_SEED)
    paths = _get_cache_paths(dataset_name)

    # 1. Attempt to load from cache
    if load_cached_data:
        # Check if essential files exist
        if os.path.exists(paths["X"]) and os.path.exists(paths["ids"]):
            # For labeled sets, check y and classes
            if dataset_name == "test" or (
                os.path.exists(paths["y"]) and os.path.exists(paths["classes"])
            ):
                print(f"Loading {dataset_name} data from cache...")
                X = np.load(paths["X"])
                ids = np.load(paths["ids"])

                y = None
                classes = None
                if dataset_name != "test":
                    y = np.load(paths["y"])
                    classes = np.load(paths["classes"], allow_pickle=True)

                return X, y, ids, classes

    # 2. Process from scratch
    print(f"Processing {dataset_name} data from scratch...")

    # Load Metadata
    metadata_path = os.path.join(METADATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    ids = df["id"].values

    # --- Feature Engineering ---

    # Part A: Global Features (192 columns)
    # Filter columns: margin_*, shape_*, texture_*
    # We assume the CSV columns are ordered correctly.
    # We explicitly drop non-feature columns.
    cols_to_drop = ["id", "image_path"]
    if "species" in df.columns:
        cols_to_drop.append("species")

    # Select feature columns
    # Note: df.drop ensures we remove metadata, leaving only features.
    # We trust the metadata generation preserved the 192 feature columns.
    X_global = df.drop(
        columns=[c for c in cols_to_drop if c in df.columns]
    ).values.astype(FLOAT_PRECISION)

    # Part B: Morphometric Features (12 columns)
    # This uses the image_processing library which handles image loading and feature extraction
    X_morph = extract_morphometrics(dataset_name, load_cached_data=load_cached_data)

    # Concatenate: [Global (192) | Morph (12)]
    # Resulting shape: (N, 204)
    X = np.hstack([X_global, X_morph]).astype(FLOAT_PRECISION)

    # --- Label Encoding ---
    y = None
    classes = None

    if "species" in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df["species"])
        classes = le.classes_  # Alphabetically sorted unique species

        # Save labels
        np.save(paths["y"], y)
        np.save(paths["classes"], classes)

    # Save features and IDs
    np.save(paths["X"], X)
    np.save(paths["ids"], ids)

    return X, y, ids, classes


def get_data_splits(load_cached_data=True):
    """
    Retrieves the Train and Validation splits for Phase 1 (Selection).

    Returns:
        tuple: (X_train, y_train, X_val, y_val, classes)
    """
    X_train, y_train, _, classes_train = load_dataset("train", load_cached_data)
    X_val, y_val, _, classes_val = load_dataset("val", load_cached_data)

    # Consistency Check: Ensure class encodings match
    # Since LabelEncoder sorts alphabetically, they match if the set of unique classes is identical.
    # EDA confirmed both sets have 99 classes.
    if not np.array_equal(classes_train, classes_val):
        print(
            "Warning: Class mismatch between Train and Val. Re-aligning Validation labels..."
        )
        # Re-map val labels to match train classes
        # This handles the edge case where Val might be missing a class or ordered differently (unlikely)

        # We need the original string labels for Val to re-map
        # Since we only have encoded 'y_val', we reconstruct strings using 'classes_val'
        y_val_strings = classes_val[y_val]

        # Create mapping from Train
        class_to_idx = {cls: i for i, cls in enumerate(classes_train)}

        # Re-encode Val
        # If a class in Val is not in Train, this will fail (as it should)
        y_val = np.array([class_to_idx[s] for s in y_val_strings])

        # Use Train classes as the master list
        classes = classes_train
    else:
        classes = classes_train

    return X_train, y_train, X_val, y_val, classes


def get_full_train_data(load_cached_data=True):
    """
    Retrieves the combined Train + Validation dataset for Phase 2 (Retraining).

    Returns:
        tuple: (X_full, y_full, classes)
    """
    # Reuse get_data_splits to ensure consistency
    X_train, y_train, X_val, y_val, classes = get_data_splits(load_cached_data)

    # Concatenate
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    return X_full, y_full, classes


def get_test_data(load_cached_data=True):
    """
    Retrieves the Test dataset for inference.

    Returns:
        tuple: (X_test, ids)
    """
    X_test, _, ids, _ = load_dataset("test", load_cached_data)
    return X_test, ids
