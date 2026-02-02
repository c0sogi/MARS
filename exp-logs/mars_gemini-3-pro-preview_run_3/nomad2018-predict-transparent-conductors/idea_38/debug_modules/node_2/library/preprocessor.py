import os
import json
import numpy as np
import pandas as pd
from library.config import WORKING_DIR


class TargetTransformer:
    """
    Applies log(1+y) transformation to targets and provides inverse transformation.
    """

    def __init__(self):
        pass

    def transform(self, y):
        """
        Apply log1p transformation.
        """
        return np.log1p(y)

    def inverse_transform(self, y_log):
        """
        Apply expm1 transformation to revert log1p.
        """
        return np.expm1(y_log)


class FeatureCleaner:
    """
    Identifies and drops constant or quasi-constant columns from the feature matrix.
    """

    def __init__(self, variance_threshold=0.0):
        self.variance_threshold = variance_threshold
        self.valid_columns = None

    def fit(self, X):
        """
        Identifies columns to keep based on variance.
        Expects a pandas DataFrame.
        """
        # Calculate variance for each column
        variances = X.var(numeric_only=True)
        # Identify columns with variance > threshold
        self.valid_columns = variances[
            variances > self.variance_threshold
        ].index.tolist()

        # Ensure we keep non-numeric columns if they exist (though descriptors are usually numeric)
        # For this specific task, descriptors are numeric.
        # If there were categorical cols, we'd need a different strategy (e.g. nunique > 1).
        return self

    def transform(self, X):
        """
        Returns dataframe with only valid columns.
        """
        if self.valid_columns is None:
            raise ValueError("FeatureCleaner has not been fitted yet.")

        # Only keep columns that are present in X (handling alignment issues if any)
        cols_to_use = [c for c in self.valid_columns if c in X.columns]
        return X[cols_to_use]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save_state(self, path):
        """
        Saves the list of valid columns to a JSON file.
        """
        if self.valid_columns is None:
            print("Warning: No state to save (not fitted).")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"valid_columns": self.valid_columns}, f)
        print(f"FeatureCleaner state saved to {path}")

    def load_state(self, path):
        """
        Loads the list of valid columns from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found at {path}")

        with open(path, "r") as f:
            data = json.load(f)
        self.valid_columns = data["valid_columns"]
        print(f"FeatureCleaner state loaded from {path}")


def preprocess_features(
    X_train, X_val, X_test, load_cached_data=True, cache_dir=WORKING_DIR
):
    """
    Applies FeatureCleaner to training data and transforms val/test data.
    Implements caching using parquet files.
    """
    # Define cache paths
    train_cache = os.path.join(cache_dir, "train_cleaned.parquet")
    val_cache = os.path.join(cache_dir, "val_cleaned.parquet")
    test_cache = os.path.join(cache_dir, "test_cleaned.parquet")
    state_cache = os.path.join(cache_dir, "preprocessor_state.json")

    # Ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(state_cache)
        ):

            print("Loading cleaned features from cache...")
            try:
                X_train_clean = pd.read_parquet(train_cache)
                X_val_clean = pd.read_parquet(val_cache)
                X_test_clean = pd.read_parquet(test_cache)

                # Load cleaner state just in case it's needed later
                cleaner = FeatureCleaner()
                cleaner.load_state(state_cache)

                return X_train_clean, X_val_clean, X_test_clean, cleaner
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print("Cache files not found. Recomputing...")

    # 2. Compute from scratch
    print("Fitting FeatureCleaner and transforming data...")
    cleaner = FeatureCleaner(variance_threshold=0.0)  # Drop strictly constant columns

    # Fit on train, transform all
    X_train_clean = cleaner.fit_transform(X_train)
    X_val_clean = cleaner.transform(X_val)
    X_test_clean = cleaner.transform(X_test)

    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Cleaned feature count:  {X_train_clean.shape[1]}")

    # 3. Save to cache
    print("Saving cleaned features to cache...")
    X_train_clean.to_parquet(train_cache)
    X_val_clean.to_parquet(val_cache)
    X_test_clean.to_parquet(test_cache)
    cleaner.save_state(state_cache)

    return X_train_clean, X_val_clean, X_test_clean, cleaner
