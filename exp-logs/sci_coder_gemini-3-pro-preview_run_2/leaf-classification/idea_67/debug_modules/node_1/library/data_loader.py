import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    CACHE_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
    TARGET_COL,
    ID_COL,
    ALL_FEATURE_COLS,
)
from library.image_features import extract_morphometrics
from library.utils import set_seed


def load_data(load_cached_data=True):
    """
    Loads training, validation, and test data. Merges provided features with
    extracted morphometric features. Handles caching and label encoding.

    Args:
        load_cached_data (bool): Whether to load merged data from cache if available.

    Returns:
        dict: A dictionary containing:
            - 'X_train' (np.ndarray): Training features (float64).
            - 'y_train' (np.ndarray): Training labels (int).
            - 'X_val' (np.ndarray): Validation features (float64).
            - 'y_val' (np.ndarray): Validation labels (int).
            - 'X_test' (np.ndarray): Test features (float64).
            - 'test_ids' (np.ndarray): Test image IDs.
            - 'classes' (np.ndarray): Array of class names corresponding to label indices.
            - 'feature_slices' (dict): Mapping of group names ('margin', 'shape', 'texture', 'morphometrics')
                                       to slice objects or indices.
            - 'feature_names' (list): List of all feature column names in order.
    """
    set_seed()

    # Define cache paths for the merged dataframes
    train_cache = os.path.join(CACHE_DIR, "train_merged.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_merged.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_merged.parquet")

    # 1. Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading merged datasets from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
    else:
        print("Processing datasets from scratch...")

        # Load metadata
        df_train_meta = pd.read_csv(TRAIN_DATA_PATH)
        df_val_meta = pd.read_csv(VAL_DATA_PATH)
        df_test_meta = pd.read_csv(TEST_DATA_PATH)

        # Extract Morphometrics (this function handles its own caching of the raw extraction)
        # We pass load_cached_data down to it.
        df_train_morph = extract_morphometrics(
            df_train_meta, "train", load_cached_data=load_cached_data
        )
        df_val_morph = extract_morphometrics(
            df_val_meta, "val", load_cached_data=load_cached_data
        )
        df_test_morph = extract_morphometrics(
            df_test_meta, "test", load_cached_data=load_cached_data
        )

        # Merge Metadata with Morphometrics on ID
        # Metadata contains [id, species, margin_*, shape_*, texture_*, image_path]
        # Morphometrics contains [id, hu_*, geometric_*]

        df_train = pd.merge(df_train_meta, df_train_morph, on=ID_COL, how="inner")
        df_val = pd.merge(df_val_meta, df_val_morph, on=ID_COL, how="inner")
        df_test = pd.merge(df_test_meta, df_test_morph, on=ID_COL, how="inner")

        # Save merged dataframes to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

    # 2. Define Feature Columns Order
    # We need to identify the morphometric columns dynamically
    # Exclude ID, Target, Image Path, and known feature columns to find morphometrics
    exclude_cols = set([ID_COL, TARGET_COL, "image_path"] + ALL_FEATURE_COLS)
    morph_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Sort morph cols to ensure consistency
    morph_cols.sort()

    # Construct the final ordered list of feature columns
    # Order: Margin -> Shape -> Texture -> Morphometrics
    final_feature_cols = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS + morph_cols

    # 3. Create Feature Slices
    # These slices will be used by the StratifiedLDAReducer and InteractionTransformer
    n_margin = len(MARGIN_COLS)
    n_shape = len(SHAPE_COLS)
    n_texture = len(TEXTURE_COLS)
    n_morph = len(morph_cols)

    current_idx = 0
    feature_slices = {}

    feature_slices["margin"] = slice(current_idx, current_idx + n_margin)
    current_idx += n_margin

    feature_slices["shape"] = slice(current_idx, current_idx + n_shape)
    current_idx += n_shape

    feature_slices["texture"] = slice(current_idx, current_idx + n_texture)
    current_idx += n_texture

    feature_slices["morphometrics"] = slice(current_idx, current_idx + n_morph)

    # 4. Prepare X matrices (float64)
    X_train = df_train[final_feature_cols].values.astype(np.float64)
    X_val = df_val[final_feature_cols].values.astype(np.float64)
    X_test = df_test[final_feature_cols].values.astype(np.float64)

    # 5. Prepare Targets
    le = LabelEncoder()
    # Fit on combined train + val species to ensure all classes are covered and indices match
    all_species = pd.concat([df_train[TARGET_COL], df_val[TARGET_COL]]).unique()
    all_species.sort()  # Sort for deterministic behavior
    le.fit(all_species)

    y_train = le.transform(df_train[TARGET_COL])
    y_val = le.transform(df_val[TARGET_COL])

    classes = le.classes_

    # 6. Prepare Test IDs
    test_ids = df_test[ID_COL].values

    # 7. Return Data Dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "test_ids": test_ids,
        "classes": classes,
        "feature_slices": feature_slices,
        "feature_names": final_feature_cols,
    }
