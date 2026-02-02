import os
import json
import numpy as np
import pandas as pd
from library.config import CACHE_DIR, RANDOM_SEED
from library.data_manager import MaterialDataset

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class DataPreprocessor:
    """
    Handles data preprocessing steps including:
    1. Log-transformation of target variables.
    2. Removal of constant (zero-variance) features.
    3. State persistence for consistent processing across train/val/test splits.
    """

    def __init__(self):
        self.cols_to_drop = []
        self.fitted = False

    @staticmethod
    def log_transform(y):
        """
        Applies log(1 + y) transformation to target variables.
        """
        return np.log1p(y)

    @staticmethod
    def inverse_log_transform(y_log):
        """
        Applies exp(y) - 1 transformation to revert log-transformed predictions.
        """
        return np.expm1(y_log)

    def fit(self, df):
        """
        Identifies constant (zero-variance) columns in the dataframe.
        """
        # Select only numeric columns for variance check
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate variance (std dev == 0 implies variance == 0)
        # Fill NaNs with 0 to avoid errors during check, though data should be clean
        stds = numeric_df.fillna(0).std()

        # Identify columns with zero standard deviation
        self.cols_to_drop = stds[stds == 0].index.tolist()

        # Also drop columns that are all NaNs (if any remain)
        nan_cols = df.columns[df.isna().all()].tolist()
        self.cols_to_drop = list(set(self.cols_to_drop + nan_cols))

        self.fitted = True
        print(
            f"Preprocessor fitted. Identified {len(self.cols_to_drop)} constant columns to drop."
        )

    def transform(self, df):
        """
        Drops the identified constant columns from the dataframe.
        """
        if not self.fitted:
            raise RuntimeError(
                "DataPreprocessor must be fitted before calling transform."
            )

        # Only drop columns that exist in the current dataframe
        cols_existing = [c for c in self.cols_to_drop if c in df.columns]
        if cols_existing:
            return df.drop(columns=cols_existing)
        return df

    def fit_transform(self, df):
        """
        Fits the preprocessor on the dataframe and returns the transformed version.
        """
        self.fit(df)
        return self.transform(df)

    def clean_features(self, df, fit=False):
        """
        Wrapper method to clean features, optionally fitting first.
        """
        if fit:
            self.fit(df)
        return self.transform(df)

    def save_state(self, path):
        """
        Saves the list of dropped columns to a JSON file.
        """
        state = {"cols_to_drop": self.cols_to_drop, "fitted": self.fitted}
        with open(path, "w") as f:
            json.dump(state, f)
        print(f"Preprocessor state saved to {path}")

    def load_state(self, path):
        """
        Loads the list of dropped columns from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found at {path}")

        with open(path, "r") as f:
            state = json.load(f)

        self.cols_to_drop = state.get("cols_to_drop", [])
        self.fitted = state.get("fitted", False)
        print(f"Preprocessor state loaded from {path}")


def get_preprocessed_dataset(split, preprocessor, load_cached_data=True):
    """
    Orchestrates loading, cleaning, and caching of datasets.

    Args:
        split (str): 'train', 'val', or 'test'.
        preprocessor (DataPreprocessor): Instance of the preprocessor.
        load_cached_data (bool): Whether to use cached files.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    # Define cache paths
    # We cache the *cleaned* dataframe
    cleaned_cache_path = os.path.join(CACHE_DIR, f"{split}_cleaned.parquet")
    # We also need to persist the preprocessor state (columns dropped) to ensure consistency
    state_path = os.path.join(CACHE_DIR, "preprocessor_state.json")

    # 1. Try to load from cache
    if load_cached_data:
        # For training, we need both the data and the state (to populate the preprocessor)
        if split == "train":
            if os.path.exists(cleaned_cache_path) and os.path.exists(state_path):
                print(
                    f"Loading cached cleaned data for {split} from {cleaned_cache_path}"
                )
                try:
                    df = pd.read_parquet(cleaned_cache_path)
                    preprocessor.load_state(state_path)
                    return df
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")

        # For val/test, we just need the data (assuming preprocessor is already fitted or we don't care about state if just loading)
        # However, correct usage implies preprocessor matches the data.
        else:
            if os.path.exists(cleaned_cache_path):
                print(
                    f"Loading cached cleaned data for {split} from {cleaned_cache_path}"
                )
                try:
                    df = pd.read_parquet(cleaned_cache_path)
                    return df
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from raw features...")

    # Load raw features using DataManager (which handles its own caching of raw extraction)
    dataset_manager = MaterialDataset()
    raw_df = dataset_manager.construct_feature_matrix(
        split, load_cached_data=load_cached_data
    )

    # Process
    if split == "train":
        # Fit on train and save state
        cleaned_df = preprocessor.fit_transform(raw_df)
        preprocessor.save_state(state_path)
    else:
        # Transform val/test using existing state
        if not preprocessor.fitted:
            # If we are here for val/test but preprocessor isn't fitted, try to load state
            if os.path.exists(state_path):
                preprocessor.load_state(state_path)
            else:
                raise RuntimeError(
                    f"Preprocessor is not fitted and no state file found at {state_path}. Run 'train' split first."
                )

        cleaned_df = preprocessor.transform(raw_df)

    # 3. Save to cache
    try:
        cleaned_df.to_parquet(cleaned_cache_path)
        print(f"Saved processed {split} data to {cleaned_cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache for {split}: {e}")

    return cleaned_df
