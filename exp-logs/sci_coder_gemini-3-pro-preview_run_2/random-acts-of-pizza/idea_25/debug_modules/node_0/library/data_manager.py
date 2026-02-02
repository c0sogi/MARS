import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("data_manager")


class DataManager:
    """
    Handles data loading, merging with metadata, preprocessing, and caching.
    """

    def __init__(self):
        pass

    def load_dataset(
        self,
        split: str = "train",
        load_cached_data: bool = True,
        debug: bool = Config.DEBUG,
    ) -> pd.DataFrame:
        """
        Loads the dataset for the specified split (train, val, test).

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from parquet cache.
            debug (bool): If True, returns a small subset of the data.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        # Determine cache path based on split
        if split == "train":
            cache_path = Config.TRAIN_FEATURES_PATH
        elif split == "val":
            cache_path = Config.VAL_FEATURES_PATH
        elif split == "test":
            cache_path = Config.TEST_FEATURES_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        # Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading {split} data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                # If debug mode is on, slice the cached data
                if debug:
                    logger.info(
                        f"Debug mode: slicing {split} data to {Config.DEV_SAMPLE_SIZE} samples."
                    )
                    df = df.head(Config.DEV_SAMPLE_SIZE)
                return df
            except Exception as e:
                logger.warning(
                    f"Failed to load cache for {split}: {e}. Reprocessing from raw data."
                )

        # Process from raw data if cache missing or load failed
        logger.info(f"Processing raw data for split: {split}")
        df = self._process_raw_data(split)

        # Save to cache (only if not in debug mode, or if we want to cache the full set before slicing)
        # We cache the FULL dataset. Slicing for debug happens after loading/processing.
        try:
            logger.info(f"Saving {split} data to cache: {cache_path}")
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save cache for {split}: {e}")

        # Handle debug slicing
        if debug:
            logger.info(
                f"Debug mode: slicing {split} data to {Config.DEV_SAMPLE_SIZE} samples."
            )
            df = df.head(Config.DEV_SAMPLE_SIZE)

        return df

    def _process_raw_data(self, split: str) -> pd.DataFrame:
        """
        Internal method to load raw JSON and CSV metadata, merge them, and clean features.
        """
        # 1. Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_META
            source_json = Config.TRAIN_JSON
        elif split == "val":
            meta_path = Config.VAL_META
            # Validation set comes from the original train.json
            source_json = Config.TRAIN_JSON
        elif split == "test":
            meta_path = Config.TEST_META
            source_json = Config.TEST_JSON
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # 2. Load Raw JSON
        # Note: For train/val, we load the same large JSON.
        # In a production env, we might optimize this, but here we follow the structure.
        if not os.path.exists(source_json):
            raise FileNotFoundError(f"Raw data file not found: {source_json}")

        with open(source_json, "r") as f:
            raw_data = json.load(f)

        df_raw = pd.DataFrame(raw_data)

        # 3. Merge
        # We use request_id as the key.
        # The metadata df contains the 'requester_received_pizza' label for train/val.
        # We perform a left join on metadata to keep only the samples for this split.
        df_merged = df_meta.merge(
            df_raw, on="request_id", how="left", suffixes=("", "_raw")
        )

        # Resolve potential column conflicts (e.g. label column in both meta and raw)
        if "requester_received_pizza_raw" in df_merged.columns:
            # If label is in metadata (it should be for train/val), we prefer that,
            # or ensure they match. Metadata is the ground truth for the split.
            df_merged.drop(columns=["requester_received_pizza_raw"], inplace=True)

        # 4. Feature Engineering / Cleaning

        # A. Text Cleaning
        # Concatenate title and text for a unified semantic representation
        # Handle NaNs by filling with empty strings
        df_merged["request_title"] = df_merged["request_title"].fillna("").astype(str)
        df_merged["request_text_edit_aware"] = (
            df_merged["request_text_edit_aware"].fillna("").astype(str)
        )

        # B. Subreddits
        # Ensure 'requester_subreddits_at_request' is a list of strings
        # If it's missing, fill with empty list
        if Config.SUBREDDIT_COL in df_merged.columns:
            df_merged[Config.SUBREDDIT_COL] = df_merged[Config.SUBREDDIT_COL].apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            # Create empty lists if column missing
            df_merged[Config.SUBREDDIT_COL] = [[] for _ in range(len(df_merged))]

        # C. Numerical Features
        # Ensure all numeric columns defined in Config exist and are float/int
        for col in Config.NUMERIC_COLS:
            if col not in df_merged.columns:
                logger.warning(
                    f"Numeric column {col} missing in raw data. Filling with 0."
                )
                df_merged[col] = 0.0
            else:
                df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce").fillna(
                    0.0
                )

        # D. Target
        # Ensure target exists for train/val
        if split in ["train", "val"]:
            if Config.TARGET_COL not in df_merged.columns:
                raise ValueError(
                    f"Target column {Config.TARGET_COL} missing in {split} set."
                )
            df_merged[Config.TARGET_COL] = df_merged[Config.TARGET_COL].astype(int)

        # Select only relevant columns to keep file size manageable
        # We keep: request_id, target (if exists), text cols, subreddit col, numeric cols
        cols_to_keep = ["request_id"]
        if split in ["train", "val"]:
            cols_to_keep.append(Config.TARGET_COL)

        cols_to_keep.extend(Config.TEXT_COLS)
        cols_to_keep.append(Config.SUBREDDIT_COL)
        cols_to_keep.extend(Config.NUMERIC_COLS)

        # Filter columns
        df_final = df_merged[cols_to_keep].copy()

        return df_final
