import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from library import config


class DualStreamPipeline:
    """
    Implements the Dual-Stream preprocessing architecture (DSPGL).

    Stream A: Parametric Gaussian Anchors (PowerTransformer / Yeo-Johnson).
              Provides a robust baseline satisfying LDA normality assumptions.

    Stream B: Constrained Non-Parametric Experts (QuantileTransformer).
              Uses n_quantiles=50 to create a regularized rank-based normalizer
              that captures non-linearities without overfitting the empirical distribution.
    """

    def __init__(self):
        # Stream A: Parametric (PowerTransformer)
        # Standardize=True ensures zero mean and unit variance
        self.pt_global = PowerTransformer(method="yeo-johnson", standardize=True)
        self.pt_macro = PowerTransformer(method="yeo-johnson", standardize=True)

        # Stream B: Constrained Non-Parametric (QuantileTransformer)
        # Strictly constrained n_quantiles to prevent memorization (Lesson 76)
        self.qt_global = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=config.N_QUANTILES,
            random_state=config.RANDOM_SEED,
        )
        self.qt_macro = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=config.N_QUANTILES,
            random_state=config.RANDOM_SEED,
        )

    def fit(self, X_global, X_macro):
        """
        Fits the transformers on the provided training data.

        Args:
            X_global (np.ndarray): Global feature matrix.
            X_macro (np.ndarray): Macro feature matrix.
        """
        # Ensure precision strictly matches config (float64)
        X_global = X_global.astype(config.FLOAT_PRECISION)
        X_macro = X_macro.astype(config.FLOAT_PRECISION)

        # Fit Stream A
        self.pt_global.fit(X_global)
        self.pt_macro.fit(X_macro)

        # Fit Stream B
        self.qt_global.fit(X_global)
        self.qt_macro.fit(X_macro)

        return self

    def transform(self, X_global, X_macro):
        """
        Transforms input data into the four expert views.

        Args:
            X_global (np.ndarray): Global feature matrix.
            X_macro (np.ndarray): Macro feature matrix.

        Returns:
            dict: Dictionary containing 'global_a', 'global_b', 'macro_a', 'macro_b'.
        """
        # Ensure precision
        X_global = X_global.astype(config.FLOAT_PRECISION)
        X_macro = X_macro.astype(config.FLOAT_PRECISION)

        return {
            "global_a": self.pt_global.transform(X_global).astype(
                config.FLOAT_PRECISION
            ),
            "global_b": self.qt_global.transform(X_global).astype(
                config.FLOAT_PRECISION
            ),
            "macro_a": self.pt_macro.transform(X_macro).astype(config.FLOAT_PRECISION),
            "macro_b": self.qt_macro.transform(X_macro).astype(config.FLOAT_PRECISION),
        }


def get_transformed_data(raw_data, load_cached_data=True):
    """
    Orchestrates the Dual-Stream transformation pipeline with caching.

    Args:
        raw_data (dict): Dictionary returned by library.data.get_data().
                         Expected keys: 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed 'train', 'val', 'test' dictionaries.
              Each sub-dictionary contains the 4 feature views + targets/ids.
    """
    # Ensure cache directory exists
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    train_cache_path = os.path.join(cache_dir, "transformed_train.npz")
    val_cache_path = os.path.join(cache_dir, "transformed_val.npz")
    test_cache_path = os.path.join(cache_dir, "transformed_test.npz")

    # 1. Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):

            print(f"Loading transformed data from {cache_dir}...")
            try:
                # Load .npz files and convert to regular dicts
                # (np.load returns a NpzFile object which requires the file to stay open)
                with np.load(train_cache_path) as f:
                    train_data = {k: f[k] for k in f.files}
                with np.load(val_cache_path) as f:
                    val_data = {k: f[k] for k in f.files}
                with np.load(test_cache_path) as f:
                    test_data = {k: f[k] for k in f.files}

                return {"train": train_data, "val": val_data, "test": test_data}

            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cache files not found. Computing transformations...")
    else:
        print("Force recompute enabled. Computing transformations...")

    # 2. Compute transformations
    print("Initializing Dual-Stream Pipeline...")

    # Unpack raw data
    # raw_data['train'] -> (X_global, X_macro, y)
    X_train_global, X_train_macro, y_train = raw_data["train"]
    X_val_global, X_val_macro, y_val = raw_data["val"]
    X_test_global, X_test_macro, test_ids = raw_data["test"]

    # Initialize Pipeline
    pipeline = DualStreamPipeline()

    # Fit on TRAINING data only
    print("Fitting pipeline on training data...")
    pipeline.fit(X_train_global, X_train_macro)

    # Transform all splits
    print("Transforming datasets...")
    train_transformed = pipeline.transform(X_train_global, X_train_macro)
    val_transformed = pipeline.transform(X_val_global, X_val_macro)
    test_transformed = pipeline.transform(X_test_global, X_test_macro)

    # Attach Targets and IDs
    train_transformed["y"] = y_train
    val_transformed["y"] = y_val
    test_transformed["ids"] = test_ids

    # 3. Save to cache
    print(f"Saving transformed data to {cache_dir}...")
    np.savez(train_cache_path, **train_transformed)
    np.savez(val_cache_path, **val_transformed)
    np.savez(test_cache_path, **test_transformed)

    return {
        "train": train_transformed,
        "val": val_transformed,
        "test": test_transformed,
    }
