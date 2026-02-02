import os
import numpy as np
import pandas as pd
from library.config import Config
from library.features import get_morphometric_features


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Constructs the 'Global' and 'Combined' feature views.
    Enforces float64 precision and implements caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays from disk.

    Returns:
        dict: A dictionary containing the datasets with the following structure:
            {
                "train": {
                    "y": np.ndarray,      # (N,) int labels
                    "ids": np.ndarray,    # (N,) int ids
                    "views": {
                        "global": np.ndarray,   # (N, 192) float64
                        "combined": np.ndarray  # (N, 203) float64
                    }
                },
                "val": { ... },
                "test": { ... }, # No 'y' key for test
                "classes": np.ndarray # (K,) string class names
            }
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "classes": "classes.npy",
        "train_y": "y_train.npy",
        "train_ids": "ids_train.npy",
        "train_global": "X_train_global.npy",
        "train_combined": "X_train_combined.npy",
        "val_y": "y_val.npy",
        "val_ids": "ids_val.npy",
        "val_global": "X_val_global.npy",
        "val_combined": "X_val_combined.npy",
        "test_ids": "ids_test.npy",
        "test_global": "X_test_global.npy",
        "test_combined": "X_test_combined.npy",
    }

    # 1. Attempt to Load from Cache
    if load_cached_data:
        all_exist = all(
            os.path.exists(os.path.join(cache_dir, f)) for f in cache_files.values()
        )
        if all_exist:
            print("Loading datasets from cache...")
            try:
                data = {
                    "classes": np.load(
                        os.path.join(cache_dir, cache_files["classes"]),
                        allow_pickle=True,
                    ),
                    "train": {
                        "y": np.load(os.path.join(cache_dir, cache_files["train_y"])),
                        "ids": np.load(
                            os.path.join(cache_dir, cache_files["train_ids"])
                        ),
                        "views": {
                            Config.VIEW_GLOBAL: np.load(
                                os.path.join(cache_dir, cache_files["train_global"])
                            ),
                            Config.VIEW_COMBINED: np.load(
                                os.path.join(cache_dir, cache_files["train_combined"])
                            ),
                        },
                    },
                    "val": {
                        "y": np.load(os.path.join(cache_dir, cache_files["val_y"])),
                        "ids": np.load(os.path.join(cache_dir, cache_files["val_ids"])),
                        "views": {
                            Config.VIEW_GLOBAL: np.load(
                                os.path.join(cache_dir, cache_files["val_global"])
                            ),
                            Config.VIEW_COMBINED: np.load(
                                os.path.join(cache_dir, cache_files["val_combined"])
                            ),
                        },
                    },
                    "test": {
                        "ids": np.load(
                            os.path.join(cache_dir, cache_files["test_ids"])
                        ),
                        "views": {
                            Config.VIEW_GLOBAL: np.load(
                                os.path.join(cache_dir, cache_files["test_global"])
                            ),
                            Config.VIEW_COMBINED: np.load(
                                os.path.join(cache_dir, cache_files["test_combined"])
                            ),
                        },
                    },
                }
                return data
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cache incomplete or missing. Recomputing...")

    # 2. Load Metadata
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # 3. Process Labels (Species)
    # We unite train and val species to ensure we capture all classes (though stratification should handle this)
    unique_species = sorted(
        pd.concat([df_train[Config.TARGET_COL], df_val[Config.TARGET_COL]]).unique()
    )
    classes = np.array(unique_species)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    y_train = df_train[Config.TARGET_COL].map(class_to_idx).values.astype(int)
    y_val = df_val[Config.TARGET_COL].map(class_to_idx).values.astype(int)

    # 4. Extract IDs
    ids_train = df_train[Config.ID_COL].values
    ids_val = df_val[Config.ID_COL].values
    ids_test = df_test[Config.ID_COL].values

    # 5. Extract Provided Features (Global View)
    # Columns starting with margin, shape, or texture
    feature_cols = [
        c for c in df_train.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Ensure strict column order
    feature_cols.sort()

    X_train_global = df_train[feature_cols].values.astype(Config.FLOAT_PRECISION)
    X_val_global = df_val[feature_cols].values.astype(Config.FLOAT_PRECISION)
    X_test_global = df_test[feature_cols].values.astype(Config.FLOAT_PRECISION)

    # 6. Extract Morphometric Features
    # This uses the library function which handles its own internal caching for the raw extraction
    X_train_morph = get_morphometric_features(
        df_train, "train", load_cached_data=load_cached_data
    )
    X_val_morph = get_morphometric_features(
        df_val, "val", load_cached_data=load_cached_data
    )
    X_test_morph = get_morphometric_features(
        df_test, "test", load_cached_data=load_cached_data
    )

    # 7. Construct Combined View (Global + Morphometric)
    X_train_combined = np.hstack([X_train_global, X_train_morph])
    X_val_combined = np.hstack([X_val_global, X_val_morph])
    X_test_combined = np.hstack([X_test_global, X_test_morph])

    # 8. Save to Cache
    print("Saving processed datasets to cache...")
    np.save(os.path.join(cache_dir, cache_files["classes"]), classes)

    np.save(os.path.join(cache_dir, cache_files["train_y"]), y_train)
    np.save(os.path.join(cache_dir, cache_files["train_ids"]), ids_train)
    np.save(os.path.join(cache_dir, cache_files["train_global"]), X_train_global)
    np.save(os.path.join(cache_dir, cache_files["train_combined"]), X_train_combined)

    np.save(os.path.join(cache_dir, cache_files["val_y"]), y_val)
    np.save(os.path.join(cache_dir, cache_files["val_ids"]), ids_val)
    np.save(os.path.join(cache_dir, cache_files["val_global"]), X_val_global)
    np.save(os.path.join(cache_dir, cache_files["val_combined"]), X_val_combined)

    np.save(os.path.join(cache_dir, cache_files["test_ids"]), ids_test)
    np.save(os.path.join(cache_dir, cache_files["test_global"]), X_test_global)
    np.save(os.path.join(cache_dir, cache_files["test_combined"]), X_test_combined)

    # 9. Return Data Structure
    data = {
        "classes": classes,
        "train": {
            "y": y_train,
            "ids": ids_train,
            "views": {
                Config.VIEW_GLOBAL: X_train_global,
                Config.VIEW_COMBINED: X_train_combined,
            },
        },
        "val": {
            "y": y_val,
            "ids": ids_val,
            "views": {
                Config.VIEW_GLOBAL: X_val_global,
                Config.VIEW_COMBINED: X_val_combined,
            },
        },
        "test": {
            "ids": ids_test,
            "views": {
                Config.VIEW_GLOBAL: X_test_global,
                Config.VIEW_COMBINED: X_test_combined,
            },
        },
    }

    return data
