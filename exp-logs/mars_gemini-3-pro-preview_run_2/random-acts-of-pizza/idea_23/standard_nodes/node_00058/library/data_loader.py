import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class DataLoader:
    """
    Handles loading and initial processing of the Reddit Pizza Request dataset.
    Merges metadata splits with raw JSON data and manages caching via Parquet files.
    """

    def __init__(self):
        self.logger = setup_logger("DataLoader")

    def load_data(self, load_cached_data: bool = True):
        """
        Loads training, validation, and test datasets.
        Uses caching to speed up subsequent runs.

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            tuple: (df_train, df_val, df_test) containing the processed DataFrames.
        """
        # Ensure working directory exists for caching
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache file paths
        train_cache = os.path.join(Config.WORKING_DIR, "train_merged.parquet")
        val_cache = os.path.join(Config.WORKING_DIR, "val_merged.parquet")
        test_cache = os.path.join(Config.WORKING_DIR, "test_merged.parquet")

        # Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(train_cache)
                and os.path.exists(val_cache)
                and os.path.exists(test_cache)
            ):
                self.logger.info("Loading datasets from cache...")
                try:
                    df_train = pd.read_parquet(train_cache)
                    df_val = pd.read_parquet(val_cache)
                    df_test = pd.read_parquet(test_cache)
                    return df_train, df_val, df_test
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load cache: {e}. Processing from scratch."
                    )
            else:
                self.logger.info("Cache files not found. Processing from scratch.")

        # Load Metadata
        self.logger.info("Loading metadata files...")
        df_meta_train = pd.read_csv(Config.TRAIN_META)
        df_meta_val = pd.read_csv(Config.VAL_META)
        df_meta_test = pd.read_csv(Config.TEST_META)

        # Apply subsampling if configured (useful for debugging/testing pipelines)
        if Config.MAX_SAMPLES is not None:
            self.logger.info(
                f"Subsampling data to {Config.MAX_SAMPLES} samples per split."
            )
            df_meta_train = df_meta_train.iloc[: Config.MAX_SAMPLES]
            df_meta_val = df_meta_val.iloc[: Config.MAX_SAMPLES]
            df_meta_test = df_meta_test.iloc[: Config.MAX_SAMPLES]

        # Load Raw JSON Data
        self.logger.info("Loading raw JSON data...")
        with open(Config.TRAIN_JSON, "r") as f:
            raw_train_data = json.load(f)

        with open(Config.TEST_JSON, "r") as f:
            raw_test_data = json.load(f)

        # Convert raw lists to DataFrames
        # Note: raw_train_data contains the full set (train + val)
        df_raw_train_full = pd.DataFrame(raw_train_data)
        df_raw_test = pd.DataFrame(raw_test_data)

        # Merge Metadata with Raw Data and Clean
        self.logger.info("Merging metadata with raw features...")
        df_train = self._process_split(df_meta_train, df_raw_train_full, is_test=False)
        df_val = self._process_split(df_meta_val, df_raw_train_full, is_test=False)
        df_test = self._process_split(df_meta_test, df_raw_test, is_test=True)

        # Save processed data to Cache
        self.logger.info("Saving processed datasets to cache...")
        try:
            df_train.to_parquet(train_cache, index=False)
            df_val.to_parquet(val_cache, index=False)
            df_test.to_parquet(test_cache, index=False)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

        return df_train, df_val, df_test

    def _process_split(
        self, meta_df: pd.DataFrame, raw_df: pd.DataFrame, is_test: bool
    ) -> pd.DataFrame:
        """
        Merges metadata with raw data, selects configured columns, and performs basic type cleaning.
        """
        # Merge on request_id
        # meta_df contains the ground truth labels for train/val splits
        # We use a left join to preserve the split definition in metadata
        merged = pd.merge(
            meta_df, raw_df, on=Config.ID_COL, how="left", suffixes=("", "_raw")
        )

        # Identify columns to keep based on Config
        cols_to_keep = [Config.ID_COL]

        # Add Target column if available and not test set
        if not is_test:
            if Config.TARGET_COL in merged.columns:
                cols_to_keep.append(Config.TARGET_COL)

        # Add Text Columns
        for col in Config.TEXT_COLS:
            if col in merged.columns:
                cols_to_keep.append(col)

        # Add Numeric Columns
        for col in Config.NUMERIC_COLS:
            if col in merged.columns:
                cols_to_keep.append(col)

        # Select relevant columns
        df_processed = merged[cols_to_keep].copy()

        # --- Data Cleaning ---

        # 1. Handle Text: Fill NaNs with empty string to ensure valid string operations later
        for col in Config.TEXT_COLS:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna("").astype(str)

        # 2. Handle Numerics: Coerce to numeric types and fill NaNs with 0
        for col in Config.NUMERIC_COLS:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(
                    df_processed[col], errors="coerce"
                ).fillna(0)

        # 3. Handle Target: Ensure integer type for classification (0 or 1)
        if not is_test and Config.TARGET_COL in df_processed.columns:
            df_processed[Config.TARGET_COL] = df_processed[Config.TARGET_COL].astype(
                int
            )

        return df_processed
