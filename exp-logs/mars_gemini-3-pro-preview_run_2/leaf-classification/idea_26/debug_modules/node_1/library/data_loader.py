import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    DTYPE,
    RAW_FEATURE_PREFIXES,
    RANDOM_SEED,
)
from library.features import get_probabilistic_features
from library.utils import set_seed


def _extract_global_features(df):
    """
    Extracts the original 192 features (margin, shape, texture) from the dataframe.
    """
    # Identify columns that start with the defined prefixes
    feature_cols = [
        c
        for c in df.columns
        if any(c.startswith(prefix) for prefix in RAW_FEATURE_PREFIXES)
    ]

    # Sort columns to ensure consistent order (though usually they are ordered in CSV)
    # The dataset description implies specific ordering, but let's trust the CSV structure.
    # We'll just take them as they appear if they match the filter.
    # However, to be safe against column permutation, we can rely on the fact
    # that pandas reads them in order.

    return df[feature_cols].values.astype(DTYPE)


def prepare_datasets(load_cached_data=True):
    """
    Loads data, generates probabilistic features, creates views, and encodes labels.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' data and the 'encoder'.
              Structure:
              {
                  'train': {
                      'y': np.ndarray,
                      'views': {
                          'Global': np.ndarray,
                          'Morph': np.ndarray,
                          'Combined': np.ndarray
                      }
                  },
                  'val': { ... },
                  'test': { ... }, # No 'y' for test
                  'classes': np.ndarray # Class names for submission
              }
    """
    set_seed(RANDOM_SEED)

    cache_file = os.path.join(WORKING_DIR, "processed_data.npz")

    # -------------------------------------------------------------------------
    # 1. Attempt to Load from Cache
    # -------------------------------------------------------------------------
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading processed datasets from cache: {cache_file}")
        try:
            with np.load(cache_file, allow_pickle=True) as data:
                # Reconstruct Dictionary Structure
                dataset = {
                    "train": {
                        "y": data["train_y"],
                        "views": {
                            "Global": data["train_global"],
                            "Morph": data["train_morph"],
                            "Combined": data["train_combined"],
                        },
                    },
                    "val": {
                        "y": data["val_y"],
                        "views": {
                            "Global": data["val_global"],
                            "Morph": data["val_morph"],
                            "Combined": data["val_combined"],
                        },
                    },
                    "test": {
                        "views": {
                            "Global": data["test_global"],
                            "Morph": data["test_morph"],
                            "Combined": data["test_combined"],
                        }
                    },
                    "classes": data["classes"],
                }
                return dataset
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing from scratch...")

    # -------------------------------------------------------------------------
    # 2. Load Metadata
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # -------------------------------------------------------------------------
    # 3. Generate/Extract Views
    # -------------------------------------------------------------------------

    # --- View A: Global (Original Features) ---
    print("Extracting Global View (192 features)...")
    X_train_global = _extract_global_features(df_train)
    X_val_global = _extract_global_features(df_val)
    X_test_global = _extract_global_features(df_test)

    # --- View B: Morph (Probabilistic Morphometric Features) ---
    print("Generating Morphological View (Probabilistic Features)...")
    # This calls the expensive feature extraction pipeline (with internal caching)
    X_train_morph = get_probabilistic_features(
        df_train, "train", load_cached_data=load_cached_data
    )
    X_val_morph = get_probabilistic_features(
        df_val, "val", load_cached_data=load_cached_data
    )
    X_test_morph = get_probabilistic_features(
        df_test, "test", load_cached_data=load_cached_data
    )

    # --- View C: Combined (Concatenation) ---
    print("Creating Combined View...")
    X_train_combined = np.hstack([X_train_global, X_train_morph])
    X_val_combined = np.hstack([X_val_global, X_val_morph])
    X_test_combined = np.hstack([X_test_global, X_test_morph])

    # -------------------------------------------------------------------------
    # 4. Process Labels
    # -------------------------------------------------------------------------
    print("Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    # -------------------------------------------------------------------------
    # 5. Save to Cache
    # -------------------------------------------------------------------------
    print(f"Saving processed datasets to {cache_file}...")
    os.makedirs(WORKING_DIR, exist_ok=True)

    np.savez(
        cache_file,
        # Train
        train_y=y_train,
        train_global=X_train_global,
        train_morph=X_train_morph,
        train_combined=X_train_combined,
        # Val
        val_y=y_val,
        val_global=X_val_global,
        val_morph=X_val_morph,
        val_combined=X_val_combined,
        # Test
        test_global=X_test_global,
        test_morph=X_test_morph,
        test_combined=X_test_combined,
        # Meta
        classes=classes,
    )

    # -------------------------------------------------------------------------
    # 6. Return Structure
    # -------------------------------------------------------------------------
    dataset = {
        "train": {
            "y": y_train,
            "views": {
                "Global": X_train_global,
                "Morph": X_train_morph,
                "Combined": X_train_combined,
            },
        },
        "val": {
            "y": y_val,
            "views": {
                "Global": X_val_global,
                "Morph": X_val_morph,
                "Combined": X_val_combined,
            },
        },
        "test": {
            "views": {
                "Global": X_test_global,
                "Morph": X_test_morph,
                "Combined": X_test_combined,
            }
        },
        "classes": classes,
    }

    return dataset
