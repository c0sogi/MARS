import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import IDEA_DIR
from library.data_loader import load_datasets


class Float64Transformer:
    """
    A preprocessing pipeline that applies Yeo-Johnson Power Transformation
    followed by Standard Scaling.

    Crucially, this class enforces np.float64 precision at every step to
    prevent numerical instability and metric floors associated with float32.
    """

    def __init__(self):
        # As per strategy: Yeo-Johnson with standardize=False,
        # followed explicitly by StandardScaler.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the PowerTransformer and StandardScaler on the provided data.

        Args:
            X (np.ndarray): Training data.
        """
        # Enforce float64 input
        X_64 = X.astype(np.float64)

        # Fit PowerTransformer
        self.pt.fit(X_64)

        # Transform data to fit the Scaler on the stabilized distribution
        X_pt = self.pt.transform(X_64)

        # Fit StandardScaler
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        X_64 = X.astype(np.float64)

        # Apply Power Transform
        X_pt = self.pt.transform(X_64)

        # Apply Standard Scaling
        X_scaled = self.scaler.transform(X_pt)

        return X_scaled.astype(np.float64)

    def fit_transform(self, X):
        """
        Fits and transforms the data in one step.
        """
        self.fit(X)
        return self.transform(X)


def process_and_cache_data(load_cached_data=True):
    """
    Orchestrates the data loading, inductive preprocessing, and caching workflow.

    1. Checks for cached transformed data in ./working/idea_27/.
    2. If not found or forced reload:
       - Loads raw data using library.data_loader.
       - Fits Float64Transformer on Training data ONLY.
       - Transforms Train, Validation, and Test sets.
       - Saves transformed arrays to disk (npy format).

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'X_train', 'y_train', 'X_val', 'y_val',
              'X_test', 'ids_test', 'classes'.
    """
    # Ensure the idea directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Define paths for the transformed feature matrices
    cache_paths = {
        "X_train": os.path.join(IDEA_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(IDEA_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(IDEA_DIR, "X_test_transformed.npy"),
    }

    # 1. Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_paths.values())
        if all_exist:
            print(f"Loading transformed features from cache at {IDEA_DIR}...")
            try:
                # Load transformed X matrices
                X_train_trans = np.load(cache_paths["X_train"])
                X_val_trans = np.load(cache_paths["X_val"])
                X_test_trans = np.load(cache_paths["X_test"])

                # Load targets and meta-info using the data_loader's own caching mechanism
                # We pass load_cached_data=True to avoid re-parsing CSVs if possible
                raw_data = load_datasets(load_cached_data=True)

                return {
                    "X_train": X_train_trans,
                    "y_train": raw_data["y_train"],
                    "X_val": X_val_trans,
                    "y_val": raw_data["y_val"],
                    "X_test": X_test_trans,
                    "ids_test": raw_data["ids_test"],
                    "classes": raw_data["classes"],
                }
            except Exception as e:
                print(f"Error loading cache: {e}. Proceeding to re-process data.")
        else:
            print("Transformed data cache not found. Proceeding to process data.")
    else:
        print("Ignoring cache. Proceeding to process data.")

    # 2. Process from scratch

    # Load raw data (features are already extracted as float64 by data_loader)
    raw_data = load_datasets(load_cached_data=True)
    X_train_raw = raw_data["X_train"]
    X_val_raw = raw_data["X_val"]
    X_test_raw = raw_data["X_test"]

    print("Initializing Float64Transformer (Yeo-Johnson + StandardScaler)...")
    transformer = Float64Transformer()

    print("Fitting transformer on Training set (Inductive Fit)...")
    transformer.fit(X_train_raw)

    print("Transforming Training, Validation, and Test sets...")
    X_train_trans = transformer.transform(X_train_raw)
    X_val_trans = transformer.transform(X_val_raw)
    X_test_trans = transformer.transform(X_test_raw)

    # 3. Save to cache
    print(f"Saving transformed datasets to {IDEA_DIR}...")
    np.save(cache_paths["X_train"], X_train_trans)
    np.save(cache_paths["X_val"], X_val_trans)
    np.save(cache_paths["X_test"], X_test_trans)

    return {
        "X_train": X_train_trans,
        "y_train": raw_data["y_train"],
        "X_val": X_val_trans,
        "y_val": raw_data["y_val"],
        "X_test": X_test_trans,
        "ids_test": raw_data["ids_test"],
        "classes": raw_data["classes"],
    }
