import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    WORKING_DIR,
    FLOAT_PRECISION,
    MORPHOMETRIC_FEATURES,
)
from library.features import extract_morphometric_features


def _get_cache_paths(view, split):
    """Generates file paths for caching X, y, and ids."""
    base = os.path.join(WORKING_DIR, f"data_{view}_{split}")
    return {
        "X": f"{base}_X.npy",
        "y": f"{base}_y.npy",
        "ids": f"{base}_ids.npy",
    }


def _save_cache(data_dict, paths):
    """Saves numpy arrays to the specified paths."""
    try:
        if data_dict["X"] is not None:
            np.save(paths["X"], data_dict["X"])
        if data_dict["y"] is not None:
            np.save(paths["y"], data_dict["y"])
        if data_dict["ids"] is not None:
            np.save(paths["ids"], data_dict["ids"])
    except Exception as e:
        print(f"Warning: Failed to save cache to {paths['X']}: {e}")


def _load_cache(paths):
    """Attempts to load numpy arrays from cache paths."""
    data = {}
    try:
        if os.path.exists(paths["X"]):
            data["X"] = np.load(paths["X"]).astype(FLOAT_PRECISION)
        else:
            return None

        if os.path.exists(paths["ids"]):
            data["ids"] = np.load(paths["ids"])
        else:
            return None

        # y is optional (e.g. for test set)
        if os.path.exists(paths["y"]):
            data["y"] = np.load(paths["y"])
        else:
            data["y"] = None

        return data
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None


def _build_feature_matrix(metadata_df, morph_df, view):
    """
    Constructs the feature matrix X based on the selected view.

    Args:
        metadata_df: DataFrame containing global features and ids.
        morph_df: DataFrame containing morphometric features and ids.
        view: 'global' or 'combined'.

    Returns:
        X (np.ndarray): The feature matrix.
    """
    # 1. Identify Global Features (exclude non-feature columns)
    exclude_cols = ["id", "species", "image_path"]
    global_cols = [c for c in metadata_df.columns if c not in exclude_cols]

    # Sort columns to ensure deterministic order
    global_cols.sort()

    # Extract Global Features
    X_global = metadata_df[global_cols].values.astype(FLOAT_PRECISION)

    if view == "global":
        return X_global

    elif view == "combined":
        # Merge morphometrics on ID to ensure alignment
        # We assume metadata_df and morph_df are aligned by row, but let's be safe
        # However, for efficiency in this controlled env, we can rely on ID sorting or direct merge

        # Create a temporary DF for merging
        df_global = metadata_df[["id"]].copy()
        # Add global features index-wise (assuming metadata_df is the reference)
        # A safer way is to merge the dataframes.

        # Let's do a pandas merge to be absolutely safe about alignment
        # Prepare global part
        df_main = metadata_df[["id"] + global_cols].copy()

        # Prepare morph part
        # morph_df has ['id'] + MORPHOMETRIC_FEATURES

        # Merge
        df_merged = pd.merge(df_main, morph_df, on="id", how="left")

        # Fill NaNs if any (though there shouldn't be)
        df_merged.fillna(0.0, inplace=True)

        # Extract all feature columns
        feat_cols = global_cols + MORPHOMETRIC_FEATURES
        X_combined = df_merged[feat_cols].values.astype(FLOAT_PRECISION)

        return X_combined

    else:
        raise ValueError(f"Unknown view: {view}")


def get_datasets(view="global", load_cached_data=True):
    """
    Main function to retrieve processed datasets.

    Args:
        view (str): 'global' (192 feats) or 'combined' (192 + morphometrics).
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_data, val_data, test_data, train_full_data, class_names)
        Each data tuple contains (X, y, ids). For test_data, y is None.
    """
    # Define splits
    splits = ["train", "val", "test"]

    # Check cache for primary splits
    cache_paths = {s: _get_cache_paths(view, s) for s in splits}

    # Also define paths for train_full
    cache_paths["train_full"] = _get_cache_paths(view, "train_full")

    # Try loading all from cache
    if load_cached_data:
        loaded_data = {}
        all_cached = True

        # Check train, val, test, train_full
        for s in splits + ["train_full"]:
            d = _load_cache(cache_paths[s])
            if d is None:
                all_cached = False
                break
            loaded_data[s] = d

        # Load classes
        classes_path = os.path.join(WORKING_DIR, "classes.npy")
        if all_cached and os.path.exists(classes_path):
            classes = np.load(classes_path, allow_pickle=True)
            return (
                (
                    loaded_data["train"]["X"],
                    loaded_data["train"]["y"],
                    loaded_data["train"]["ids"],
                ),
                (
                    loaded_data["val"]["X"],
                    loaded_data["val"]["y"],
                    loaded_data["val"]["ids"],
                ),
                (loaded_data["test"]["X"], None, loaded_data["test"]["ids"]),
                (
                    loaded_data["train_full"]["X"],
                    loaded_data["train_full"]["y"],
                    loaded_data["train_full"]["ids"],
                ),
                classes,
            )

    # If not cached, compute from scratch
    # 1. Load Metadata
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)
    df_test = pd.read_csv(TEST_FILE)

    # 2. Extract Morphometrics (cached internally by library.features)
    # We need these for 'combined' view, but also to maintain consistency if we switch views later
    # The function calls are cheap if cached.
    morph_train = extract_morphometric_features(df_train, "train", load_cached_data)
    morph_val = extract_morphometric_features(df_val, "val", load_cached_data)
    morph_test = extract_morphometric_features(df_test, "test", load_cached_data)

    # 3. Encode Labels
    # Fit on all available training labels (train + val)
    all_species = pd.concat([df_train["species"], df_val["species"]]).unique()
    all_species.sort()  # Deterministic order
    le = LabelEncoder()
    le.fit(all_species)
    classes = le.classes_

    # Save classes
    np.save(os.path.join(WORKING_DIR, "classes.npy"), classes)

    # 4. Build Matrices
    datasets = {}

    # Helper to process a single split
    def process_split(df_meta, df_morph, is_test=False):
        X = _build_feature_matrix(df_meta, df_morph, view)
        ids = df_meta["id"].values
        y = None
        if not is_test:
            y = le.transform(df_meta["species"])
        return {"X": X, "y": y, "ids": ids}

    datasets["train"] = process_split(df_train, morph_train)
    datasets["val"] = process_split(df_val, morph_val)
    datasets["test"] = process_split(df_test, morph_test, is_test=True)

    # 5. Create Full Train (Train + Val)
    # Stack X and y, concatenate ids
    datasets["train_full"] = {
        "X": np.vstack((datasets["train"]["X"], datasets["val"]["X"])),
        "y": np.concatenate((datasets["train"]["y"], datasets["val"]["y"])),
        "ids": np.concatenate((datasets["train"]["ids"], datasets["val"]["ids"])),
    }

    # 6. Save to Cache
    for s in splits + ["train_full"]:
        _save_cache(datasets[s], cache_paths[s])

    return (
        (datasets["train"]["X"], datasets["train"]["y"], datasets["train"]["ids"]),
        (datasets["val"]["X"], datasets["val"]["y"], datasets["val"]["ids"]),
        (datasets["test"]["X"], None, datasets["test"]["ids"]),
        (
            datasets["train_full"]["X"],
            datasets["train_full"]["y"],
            datasets["train_full"]["ids"],
        ),
        classes,
    )
