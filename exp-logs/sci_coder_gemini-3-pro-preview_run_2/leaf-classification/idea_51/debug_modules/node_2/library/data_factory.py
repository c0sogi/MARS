import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    DTYPE,
    RANDOM_SEED,
    FEATURE_GROUPS,
    N_FEATURES_PER_GROUP,
)
from library.utils import set_seed
from library.image_features import get_morphometric_features


def load_data(load_cached_data=True):
    """
    Loads the dataset for the DSPGE strategy.

    Constructs two feature views:
    1. Global View: The 192 provided features (Margin, Shape, Texture).
    2. Morphometric View: Extracted physical features (Hu Moments, Geometry).

    Handles Label Encoding for the target variable.

    Args:
        load_cached_data (bool): If True, attempts to load processed matrices from disk.

    Returns:
        dict: A dictionary containing data for 'train', 'val', and 'test' splits,
              plus the 'classes' array.
              Structure:
              {
                  "train": {
                      "X_global": np.ndarray,
                      "X_morph": np.ndarray,
                      "y": np.ndarray,
                      "ids": np.ndarray
                  },
                  "val": { ... },
                  "test": { ... },
                  "classes": np.ndarray
              }
    """
    set_seed(RANDOM_SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache filenames
    splits = ["train", "val", "test"]
    cache_files = {}
    for split in splits:
        cache_files[split] = {
            "X_global": os.path.join(WORKING_DIR, f"{split}_X_global.npy"),
            "X_morph": os.path.join(WORKING_DIR, f"{split}_X_morph.npy"),
            "y": os.path.join(WORKING_DIR, f"{split}_y.npy"),
            "ids": os.path.join(WORKING_DIR, f"{split}_ids.npy"),
        }
    cache_classes = os.path.join(WORKING_DIR, "classes.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        all_exist = os.path.exists(cache_classes)
        for split in splits:
            for key, path in cache_files[split].items():
                # Test set doesn't have 'y', so skip checking that file for test
                if split == "test" and key == "y":
                    continue
                if not os.path.exists(path):
                    all_exist = False
                    break

        if all_exist:
            print("Loading processed data from cache...")
            data = {}
            data["classes"] = np.load(cache_classes, allow_pickle=True)

            for split in splits:
                data[split] = {}
                data[split]["X_global"] = np.load(cache_files[split]["X_global"])
                data[split]["X_morph"] = np.load(cache_files[split]["X_morph"])
                data[split]["ids"] = np.load(cache_files[split]["ids"])
                if split != "test":
                    data[split]["y"] = np.load(cache_files[split]["y"])
                else:
                    data[split]["y"] = None
            return data
        else:
            print("Cache incomplete or missing. Processing from scratch...")
    else:
        print("Forced reprocessing. Ignoring cache...")

    # 2. Process Data from Scratch

    # A. Load Morphometric Features (External Module)
    # This handles its own caching for the raw extraction part
    print("Retrieving morphometric features...")
    morph_data = get_morphometric_features(load_cached_data=load_cached_data)

    # B. Load Metadata and Global Features
    data = {}

    # Helper to generate column names for global features
    # margin_1..64, shape_1..64, texture_1..64
    global_cols = []
    for group in FEATURE_GROUPS:
        for i in range(1, N_FEATURES_PER_GROUP + 1):
            global_cols.append(f"{group}_{i}")

    # Initialize Label Encoder
    le = LabelEncoder()

    for split in splits:
        print(f"Processing {split} split...")
        metadata_path = os.path.join(METADATA_DIR, f"{split}.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file missing: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Extract Global Features
        X_global = df[global_cols].values.astype(DTYPE)

        # Extract IDs
        ids = df["id"].values

        # Retrieve Morphometric Features
        # Ensure alignment by checking IDs match
        X_morph_raw, ids_morph = morph_data[split]

        # Strict alignment check
        if not np.array_equal(ids, ids_morph):
            raise ValueError(
                f"ID mismatch between metadata and morphometric features for {split} set."
            )

        X_morph = X_morph_raw.astype(DTYPE)

        # Extract Targets (if available)
        y = None
        if "species" in df.columns:
            raw_labels = df["species"].values
            if split == "train":
                y = le.fit_transform(raw_labels)
            else:
                # Handle potential unseen labels in val (unlikely given stratification, but good practice)
                # We assume val classes are subset of train classes based on problem description
                y = le.transform(raw_labels)
            y = y.astype(int)

        # Store in dictionary
        data[split] = {"X_global": X_global, "X_morph": X_morph, "y": y, "ids": ids}

        # Save to cache
        np.save(cache_files[split]["X_global"], X_global)
        np.save(cache_files[split]["X_morph"], X_morph)
        np.save(cache_files[split]["ids"], ids)
        if y is not None:
            np.save(cache_files[split]["y"], y)

    # Save classes
    classes = le.classes_
    data["classes"] = classes
    np.save(cache_classes, classes)

    print("Data processing complete and cached.")
    return data
