import os
import json
import numpy as np
import pandas as pd
from library.config import WORKING_DIR, TARGET_COLS
from library.feature_engineering import process_split


class DataCleaner:
    """
    Handles feature cleaning by removing constant columns and filling missing values.
    Persists state (columns to drop, fill values) to ensure consistency between train/val/test.
    """

    def __init__(self):
        self.cols_to_drop = []
        self.fill_values = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the cleaner on the training data.
        Identifies constant columns and calculates median values for imputation.
        """
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude targets and id from cleaning logic (though they shouldn't be dropped anyway)
        exclude = TARGET_COLS + ["id"]
        features = [c for c in numeric_cols if c not in exclude]

        # 1. Identify Constant columns (std == 0)
        # We handle NaNs by ignoring them in std calculation
        std = df[features].std()
        self.cols_to_drop = std[std == 0].index.tolist()

        # 2. Calculate Fill values (median)
        # We compute median for all features, even those we might drop (simpler logic)
        self.fill_values = df[features].median().to_dict()

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the cleaning transformations to a dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("DataCleaner must be fitted before transform.")

        df_clean = df.copy()

        # Drop identified constant columns
        # Only drop if they exist in the current dataframe
        existing_drop = [c for c in self.cols_to_drop if c in df_clean.columns]
        if existing_drop:
            df_clean.drop(columns=existing_drop, inplace=True)

        # Fill NaNs with learned medians
        for col, val in self.fill_values.items():
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(val)

        # Fill any remaining NaNs (e.g., columns not in train but in test) with 0
        # This acts as a safety net
        df_clean.fillna(0, inplace=True)

        return df_clean

    def save_state(self, path: str):
        """Saves the cleaner state to a JSON file."""
        # Convert numpy types to python types for JSON serialization
        fill_values_serializable = {k: float(v) for k, v in self.fill_values.items()}
        state = {
            "cols_to_drop": self.cols_to_drop,
            "fill_values": fill_values_serializable,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_state(self, path: str):
        """Loads the cleaner state from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cleaner state file not found: {path}")

        with open(path, "r") as f:
            state = json.load(f)
        self.cols_to_drop = state["cols_to_drop"]
        self.fill_values = state["fill_values"]
        self.is_fitted = True
        return self


def clean_features(df: pd.DataFrame, cleaner: DataCleaner = None) -> pd.DataFrame:
    """
    Wrapper function to clean features using a DataCleaner instance.
    If cleaner is None, it assumes fitting is required (not recommended for pipelines).
    """
    if cleaner is None:
        cleaner = DataCleaner()
        cleaner.fit(df)
    return cleaner.transform(df)


def transform_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log1p transformation to the target columns.
    z = log(1 + y)
    """
    df_trans = df.copy()
    for col in TARGET_COLS:
        if col in df_trans.columns:
            # Ensure no negative values below -1 (formation energy min is 0, so safe)
            df_trans[col] = np.log1p(df_trans[col])
    return df_trans


def inverse_transform_targets(pred_array: np.ndarray) -> np.ndarray:
    """
    Applies expm1 transformation to revert predictions to original scale.
    y = exp(z) - 1
    """
    return np.expm1(pred_array)


def get_preprocessed_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main entry point to retrieve preprocessed data for a specific split.
    Handles feature generation, cleaning, target transformation, and caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed and cleaned dataframe ready for training/inference.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    clean_cache_path = os.path.join(WORKING_DIR, f"{split}_cleaned_idea_40.parquet")
    cleaner_state_path = os.path.join(WORKING_DIR, "cleaner_state_idea_40.json")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(clean_cache_path):
        # For 'train', we must ensure the cleaner state also exists to ensure reproducibility for val/test
        if split == "train" and not os.path.exists(cleaner_state_path):
            print(f"Cleaner state missing for {split}, recomputing from scratch...")
        else:
            print(f"Loading cached cleaned data for {split} from {clean_cache_path}")
            return pd.read_parquet(clean_cache_path)

    # 2. Compute from scratch
    print(f"Preprocessing {split} data from scratch...")

    # Step A: Get raw features (delegates to feature_engineering module)
    # This handles the heavy lifting of geometric analysis
    raw_df = process_split(split, load_cached_data=load_cached_data)

    cleaner = DataCleaner()

    if split == "train":
        # Fit cleaner on training data
        print("Fitting DataCleaner on training data...")
        cleaner.fit(raw_df)
        cleaner.save_state(cleaner_state_path)

        # Transform features
        df_clean = cleaner.transform(raw_df)

        # Transform targets (log1p)
        df_clean = transform_targets(df_clean)

    else:
        # Load fitted cleaner state
        print(f"Loading DataCleaner state for {split}...")
        if not os.path.exists(cleaner_state_path):
            raise RuntimeError(
                f"Cleaner state not found at {cleaner_state_path}. "
                "You must run the 'train' split first to fit the cleaner."
            )

        cleaner.load_state(cleaner_state_path)

        # Transform features
        df_clean = cleaner.transform(raw_df)

        # Transform targets if they exist (e.g. validation set)
        if all(col in df_clean.columns for col in TARGET_COLS):
            df_clean = transform_targets(df_clean)

    # Step B: Save to cache
    print(f"Saving cleaned data to {clean_cache_path}")
    df_clean.to_parquet(clean_cache_path, index=False)

    return df_clean
