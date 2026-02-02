import os
import json
import pandas as pd
from library.config import Config


class DataLoader:
    """
    Handles loading and merging of raw data with metadata splits.
    Implements lazy loading for raw JSON files to optimize performance.
    """

    def __init__(self):
        self._raw_train = None
        self._raw_test = None

    def _load_raw_json(self, path: str) -> pd.DataFrame:
        """
        Loads a JSON file into a pandas DataFrame.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _get_raw_train(self) -> pd.DataFrame:
        """
        Returns the raw training dataframe, loading it if necessary.
        """
        if self._raw_train is None:
            self._raw_train = self._load_raw_json(Config.TRAIN_JSON)
        return self._raw_train

    def _get_raw_test(self) -> pd.DataFrame:
        """
        Returns the raw test dataframe, loading it if necessary.
        """
        if self._raw_test is None:
            self._raw_test = self._load_raw_json(Config.TEST_JSON)
        return self._raw_test

    def load_split(self, split: str, sample_size: int = None) -> pd.DataFrame:
        """
        Loads the data for a specific split ('train', 'val', 'test').

        Args:
            split (str): One of 'train', 'val', 'test'.
            sample_size (int, optional): If provided, returns only the first N rows.
                                         Useful for debugging.

        Returns:
            pd.DataFrame: The merged dataframe containing features and labels (if applicable).
        """
        # Determine paths and source data based on split
        if split == "train":
            meta_path = Config.TRAIN_META
            raw_df = self._get_raw_train()
        elif split == "val":
            meta_path = Config.VAL_META
            raw_df = self._get_raw_train()
        elif split == "test":
            meta_path = Config.TEST_META
            raw_df = self._get_raw_test()
        else:
            raise ValueError(
                f"Invalid split name: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Load Metadata
        df_meta = pd.read_csv(meta_path)

        # Merge with Raw Data
        # We use a left merge on metadata to preserve the split definition.
        # Suffixes handle potential column name collisions (though we prioritize metadata).
        df_merged = df_meta.merge(
            raw_df, on="request_id", how="left", suffixes=("", "_raw")
        )

        # Define columns to keep based on Config
        cols_to_keep = ["request_id"]

        # Text Features
        cols_to_keep.append(Config.TEXT_COL_TITLE)
        cols_to_keep.append(Config.TEXT_COL_BODY)

        # Numeric Features
        cols_to_keep.extend(Config.NUMERIC_COLS)

        # Target Label (only for train/val)
        target_col = "requester_received_pizza"
        if target_col in df_merged.columns:
            cols_to_keep.append(target_col)

        # Validate column existence
        # We allow the target to be missing for the test split
        missing_cols = [c for c in cols_to_keep if c not in df_merged.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected columns in merged data for split '{split}': {missing_cols}"
            )

        # Select and copy final dataframe
        final_df = df_merged[cols_to_keep].copy()

        # Ensure target is integer if present
        if target_col in final_df.columns:
            final_df[target_col] = final_df[target_col].astype(int)

        # Apply debugging sample size if requested
        if sample_size is not None:
            final_df = final_df.head(sample_size)

        return final_df
