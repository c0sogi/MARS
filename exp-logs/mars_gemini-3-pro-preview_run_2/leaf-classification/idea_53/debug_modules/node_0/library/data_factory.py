import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
    TARGET_COL,
    ID_COL,
    MAX_TRAIN_SAMPLES,
)
from library.features import process_images


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, extracting features and structuring them for the ensemble.

    Args:
        load_cached_data (bool): If True, attempts to load processed numpy arrays from disk.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' dictionaries and 'classes'.
              Each split contains 'y' (labels) and feature views:
              'global', 'margin', 'shape', 'texture', 'morph'.
    """
    cache_file = os.path.join(WORKING_DIR, "dataset_cache.npz")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading dataset from cache: {cache_file}")
        try:
            with np.load(cache_file, allow_pickle=True) as data:
                dataset = {
                    "train": {
                        "y": data["train_y"],
                        "global": data["train_global"],
                        "margin": data["train_margin"],
                        "shape": data["train_shape"],
                        "texture": data["train_texture"],
                        "morph": data["train_morph"],
                    },
                    "val": {
                        "y": data["val_y"],
                        "global": data["val_global"],
                        "margin": data["val_margin"],
                        "shape": data["val_shape"],
                        "texture": data["val_texture"],
                        "morph": data["val_morph"],
                    },
                    "test": {
                        "global": data["test_global"],
                        "margin": data["test_margin"],
                        "shape": data["test_shape"],
                        "texture": data["test_texture"],
                        "morph": data["test_morph"],
                        "ids": data["test_ids"],  # Helper for submission
                    },
                    "classes": data["classes"],
                }
            return dataset
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing dataset...")

    # 2. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # Debugging: Subsample training data if configured
    if MAX_TRAIN_SAMPLES is not None:
        print(f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples.")
        df_train = df_train.iloc[:MAX_TRAIN_SAMPLES].copy()

    # 3. Extract Morphometrics (Image Features)
    # process_images handles its own caching of the extraction process
    print("Processing images for morphometric features...")
    morph_train = process_images(df_train, load_cached_data=load_cached_data)
    morph_val = process_images(df_val, load_cached_data=load_cached_data)
    morph_test = process_images(df_test, load_cached_data=load_cached_data)

    # 4. Merge Morphometrics with Tabular Data
    # We merge on ID to ensure alignment
    df_train = df_train.merge(morph_train, on=ID_COL, how="left")
    df_val = df_val.merge(morph_val, on=ID_COL, how="left")
    df_test = df_test.merge(morph_test, on=ID_COL, how="left")

    # Identify Morphometric Columns (exclude ID and original columns)
    # The morph_train dataframe has ID and the new features.
    morph_cols = [c for c in morph_train.columns if c != ID_COL]

    # Fill NaNs in morphometrics (e.g., if image load failed) with 0
    # This ensures the pipeline doesn't crash on missing physical features.
    for df in [df_train, df_val, df_test]:
        df[morph_cols] = df[morph_cols].fillna(0.0)

    # 5. Prepare Feature Views
    def extract_views(df):
        # Cast to float64 for precision
        v_margin = df[MARGIN_COLS].values.astype(np.float64)
        v_shape = df[SHAPE_COLS].values.astype(np.float64)
        v_texture = df[TEXTURE_COLS].values.astype(np.float64)
        v_morph = df[morph_cols].values.astype(np.float64)

        # Global view is concatenation of provided features (Margin + Shape + Texture)
        # We do NOT include morphometrics in the "Global" view defined for Topologies A/B/D
        # based on the idea description (which refers to the 192 provided features).
        v_global = np.hstack([v_margin, v_shape, v_texture])

        return v_global, v_margin, v_shape, v_texture, v_morph

    print("Structuring feature views...")
    train_global, train_margin, train_shape, train_texture, train_morph = extract_views(
        df_train
    )
    val_global, val_margin, val_shape, val_texture, val_morph = extract_views(df_val)
    test_global, test_margin, test_shape, test_texture, test_morph = extract_views(
        df_test
    )

    # 6. Encode Labels
    le = LabelEncoder()
    # Fit on all unique classes in train (stratified split ensures coverage, but safe to check)
    le.fit(df_train[TARGET_COL].unique())

    y_train = le.transform(df_train[TARGET_COL])
    y_val = le.transform(df_val[TARGET_COL])
    classes = le.classes_

    # 7. Construct Dataset Dictionary
    dataset = {
        "train": {
            "y": y_train,
            "global": train_global,
            "margin": train_margin,
            "shape": train_shape,
            "texture": train_texture,
            "morph": train_morph,
        },
        "val": {
            "y": y_val,
            "global": val_global,
            "margin": val_margin,
            "shape": val_shape,
            "texture": val_texture,
            "morph": val_morph,
        },
        "test": {
            "global": test_global,
            "margin": test_margin,
            "shape": test_shape,
            "texture": test_texture,
            "morph": test_morph,
            "ids": df_test[ID_COL].values,
        },
        "classes": classes,
    }

    # 8. Save to Cache
    print(f"Saving dataset to cache: {cache_file}")
    os.makedirs(WORKING_DIR, exist_ok=True)
    np.savez_compressed(
        cache_file,
        train_y=dataset["train"]["y"],
        train_global=dataset["train"]["global"],
        train_margin=dataset["train"]["margin"],
        train_shape=dataset["train"]["shape"],
        train_texture=dataset["train"]["texture"],
        train_morph=dataset["train"]["morph"],
        val_y=dataset["val"]["y"],
        val_global=dataset["val"]["global"],
        val_margin=dataset["val"]["margin"],
        val_shape=dataset["val"]["shape"],
        val_texture=dataset["val"]["texture"],
        val_morph=dataset["val"]["morph"],
        test_global=dataset["test"]["global"],
        test_margin=dataset["test"]["margin"],
        test_shape=dataset["test"]["shape"],
        test_texture=dataset["test"]["texture"],
        test_morph=dataset["test"]["morph"],
        test_ids=dataset["test"]["ids"],
        classes=dataset["classes"],
    )

    return dataset
