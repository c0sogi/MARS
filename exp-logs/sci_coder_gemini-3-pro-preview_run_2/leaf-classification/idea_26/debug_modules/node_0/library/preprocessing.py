import os
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from library.config import WORKING_DIR, DTYPE


def get_preprocessor():
    """
    Returns a Scikit-Learn Pipeline containing PowerTransformer(method='yeo-johnson').

    The transformer is configured with standardize=True to ensure zero mean and unit variance,
    which is optimal for the downstream Linear and Quadratic Discriminant Analysis models.

    Returns:
        Pipeline: A sklearn pipeline with the power transformer.
    """
    return Pipeline([("pt", PowerTransformer(method="yeo-johnson", standardize=True))])


def preprocess_data(dataset, load_cached_data=True):
    """
    Applies the preprocessor to the dataset views.
    Fits on the training set and transforms training, validation, and test sets.
    Implements caching to disk to avoid re-computation.

    Args:
        dataset (dict): The dictionary returned by data_loader.prepare_datasets.
                        Expected to contain 'train', 'val', 'test' keys with 'views'.
                        Each 'views' dict should contain 'Global', 'Morph', and 'Combined'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary with the same structure as input but with transformed features.
    """
    cache_path = os.path.join(WORKING_DIR, "preprocessed_data.npz")

    # -------------------------------------------------------------------------
    # 1. Attempt to Load from Cache
    # -------------------------------------------------------------------------
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading preprocessed data from cache: {cache_path}")
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                # Reconstruct Dictionary Structure
                processed_dataset = {
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
                return processed_dataset
        except Exception as e:
            print(
                f"Failed to load preprocessed cache ({e}). Recomputing from scratch..."
            )

    # -------------------------------------------------------------------------
    # 2. Compute Transformations
    # -------------------------------------------------------------------------
    print("Preprocessing data with PowerTransformer (Yeo-Johnson)...")

    # Initialize new structure to hold processed data
    # We copy y and classes directly from the input dataset
    processed_dataset = {
        "train": {"y": dataset["train"]["y"], "views": {}},
        "val": {"y": dataset["val"]["y"], "views": {}},
        "test": {"views": {}},
        "classes": dataset["classes"],
    }

    # Iterate over each view defined in the dataset
    # We assume all splits have the same views (Global, Morph, Combined)
    views = dataset["train"]["views"].keys()

    for view_name in views:
        print(f"  Transforming view: {view_name}")

        # Retrieve raw data
        X_train_raw = dataset["train"]["views"][view_name]
        X_val_raw = dataset["val"]["views"][view_name]
        X_test_raw = dataset["test"]["views"][view_name]

        # Get preprocessor
        preprocessor = get_preprocessor()

        # Fit on Training Data ONLY to prevent leakage
        preprocessor.fit(X_train_raw)

        # Transform all splits
        # Cast to DTYPE (float64) to ensure precision for density estimation
        X_train_proc = preprocessor.transform(X_train_raw).astype(DTYPE)
        X_val_proc = preprocessor.transform(X_val_raw).astype(DTYPE)
        X_test_proc = preprocessor.transform(X_test_raw).astype(DTYPE)

        # Store in new structure
        processed_dataset["train"]["views"][view_name] = X_train_proc
        processed_dataset["val"]["views"][view_name] = X_val_proc
        processed_dataset["test"]["views"][view_name] = X_test_proc

    # -------------------------------------------------------------------------
    # 3. Save to Cache
    # -------------------------------------------------------------------------
    print(f"Saving preprocessed data to {cache_path}...")
    try:
        np.savez(
            cache_path,
            # Train
            train_y=processed_dataset["train"]["y"],
            train_global=processed_dataset["train"]["views"]["Global"],
            train_morph=processed_dataset["train"]["views"]["Morph"],
            train_combined=processed_dataset["train"]["views"]["Combined"],
            # Val
            val_y=processed_dataset["val"]["y"],
            val_global=processed_dataset["val"]["views"]["Global"],
            val_morph=processed_dataset["val"]["views"]["Morph"],
            val_combined=processed_dataset["val"]["views"]["Combined"],
            # Test
            test_global=processed_dataset["test"]["views"]["Global"],
            test_morph=processed_dataset["test"]["views"]["Morph"],
            test_combined=processed_dataset["test"]["views"]["Combined"],
            # Meta
            classes=processed_dataset["classes"],
        )
    except Exception as e:
        print(f"Warning: Could not save preprocessed cache: {e}")

    return processed_dataset
