import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.image_processing import extract_morphometrics

# Constants
CACHE_DIR = "./working/idea_63"
METADATA_DIR = "./metadata"


def get_feature_groups(feature_names):
    """
    Identifies column indices for Margin, Shape, Texture, and Morphometric features.

    Args:
        feature_names (list): List of column names from the feature DataFrame.

    Returns:
        dict: Mapping of group name ('margin', 'shape', 'texture', 'morphometrics')
              to a list of integer indices.
    """
    groups = {"margin": [], "shape": [], "texture": [], "morphometrics": []}

    for idx, col in enumerate(feature_names):
        # We use string containment to identify groups.
        # Original features: 'margin_1', 'shape_1', 'texture_1', etc.
        # New features: 'morph_0', 'morph_1', etc.
        if "margin" in col:
            groups["margin"].append(idx)
        elif "shape" in col:
            groups["shape"].append(idx)
        elif "texture" in col:
            groups["texture"].append(idx)
        elif "morph" in col:
            groups["morphometrics"].append(idx)

    return groups


def _process_subset(mode, load_cached_data):
    """
    Internal function to process a single data split (train, val, or test).
    Handles loading metadata, extracting morphometrics, merging, and caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"data_{mode}.parquet")

    # 1. Try to load merged data from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{mode}] Loading merged dataset from {cache_path}...")
        try:
            df_merged = pd.read_parquet(cache_path)

            # Separate X and y/id
            if mode == "test":
                # For test, we need ID and features
                ids = df_merged["id"].values
                X = df_merged.drop(columns=["id"])
                return X, ids
            else:
                # For train/val, we need species and features
                y = df_merged["species"].values
                X = df_merged.drop(columns=["species"])
                return X, y
        except Exception as e:
            print(f"[{mode}] Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"[{mode}] Processing dataset from metadata...")

    # Load metadata
    metadata_path = os.path.join(METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Extract Morphometrics (this function handles its own caching of the raw matrix)
    # We pass 'mode' as cache_name to reuse the morphometrics cache if available
    X_morph_raw = extract_morphometrics(
        df_meta, load_cached_data=load_cached_data, cache_name=mode
    )

    # Create DataFrame for morphometrics
    morph_cols = [f"morph_{i}" for i in range(X_morph_raw.shape[1])]
    df_morph = pd.DataFrame(X_morph_raw, columns=morph_cols)

    # Select original feature columns (exclude meta columns)
    # Meta columns in train/val: id, species, image_path
    # Meta columns in test: id, image_path
    drop_cols = ["id", "image_path"]
    if "species" in df_meta.columns:
        drop_cols.append("species")

    df_features_original = df_meta.drop(
        columns=[c for c in drop_cols if c in df_meta.columns]
    )

    # Concatenate original features with morphometrics
    # Reset indices to ensure correct alignment
    df_features_original = df_features_original.reset_index(drop=True)
    df_morph = df_morph.reset_index(drop=True)

    X_combined = pd.concat([df_features_original, df_morph], axis=1)

    # Ensure float64 precision
    X_combined = X_combined.astype(np.float64)

    # Prepare DataFrame for saving (include target/id for the cache file)
    df_to_save = X_combined.copy()

    if mode == "test":
        ids = df_meta["id"].values
        df_to_save["id"] = ids
        # Save
        df_to_save.to_parquet(cache_path, index=False)
        return X_combined, ids
    else:
        y = df_meta["species"].values
        df_to_save["species"] = y
        # Save
        df_to_save.to_parquet(cache_path, index=False)
        return X_combined, y


def load_data(load_cached_data=True):
    """
    Loads the complete dataset (train, val, test).
    Encodes the target labels.

    Args:
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
               X_* are DataFrames with float64 features.
               y_* are numpy arrays of integer class indices.
               test_ids is a numpy array of image IDs.
               classes is a list/array of class names corresponding to indices.
    """
    print("Loading Data...")

    # Process subsets
    X_train, y_train_raw = _process_subset("train", load_cached_data)
    X_val, y_val_raw = _process_subset("val", load_cached_data)
    X_test, test_ids = _process_subset("test", load_cached_data)

    # Label Encoding
    # We combine train and val labels to ensure the encoder covers all classes seen in development
    all_species = np.unique(np.concatenate([y_train_raw, y_val_raw]))

    le = LabelEncoder()
    le.fit(all_species)

    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    classes = le.classes_

    print(f"Data Loaded Successfully.")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape:   {X_val.shape}")
    print(f"  Test shape:  {X_test.shape}")
    print(f"  Num Classes: {len(classes)}")

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
