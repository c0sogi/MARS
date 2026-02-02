import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import save_to_cache, load_from_cache, setup_logger

# Setup logger
logger = setup_logger(
    "data_loader", os.path.join(Config.WORKING_DIR, "data_loader.log")
)


class DataLoader:
    def __init__(self):
        self.train_path = Config.TRAIN_JSON
        self.test_path = Config.TEST_JSON
        self.metadata_map = {
            "train": Config.TRAIN_META,
            "val": Config.VAL_META,
            "test": Config.TEST_META,
        }

    def _load_raw_json(self, path):
        """Helper to load raw JSON data."""
        with open(path, "r") as f:
            return json.load(f)

    def load_dataset(self, split: str, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Loads the dataset for a specific split (train, val, test).

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: Processed dataframe with text, metadata, and targets.
        """
        if split not in self.metadata_map:
            raise ValueError(
                f"Invalid split: {split}. Must be one of {list(self.metadata_map.keys())}"
            )

        cache_path = os.path.join(Config.WORKING_DIR, f"{split}_processed.parquet")

        # 1. Try Loading from Cache
        if load_cached_data:
            cached_df = load_from_cache(cache_path)
            if cached_df is not None:
                logger.info(f"Loaded {split} data from cache: {cache_path}")
                return cached_df

        logger.info(f"Processing {split} data from scratch...")

        # 2. Load Metadata
        meta_path = self.metadata_map[split]
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # 3. Load Raw Data
        # Determine which raw file to load based on the 'source_file' column in metadata
        # We assume all records in a split come from the same source file (train.json or test.json)
        # as per the metadata generation logic.
        source_file_rel = df_meta["source_file"].iloc[0]
        source_file_abs = os.path.join(
            Config.INPUT_DIR, os.path.basename(source_file_rel)
        )

        raw_data_list = self._load_raw_json(source_file_abs)

        # 4. Extract Records using sample_index
        # Using list comprehension with index access is O(N) and efficient
        indices = df_meta["sample_index"].values
        selected_records = [raw_data_list[i] for i in indices]

        # Create DataFrame from selected records
        df_raw = pd.DataFrame(selected_records)

        # 5. Construct Final DataFrame
        df_processed = pd.DataFrame()
        df_processed[Config.ID_COL] = df_meta[Config.ID_COL]

        # Add Target if available (train/val)
        if Config.TARGET_COL in df_meta.columns:
            df_processed[Config.TARGET_COL] = df_meta[Config.TARGET_COL].astype(int)

        # Feature Engineering: Text
        # Concatenate title and body
        title_col = "request_title"
        body_col = "request_text_edit_aware"

        # Fill NaNs with empty string just in case
        titles = df_raw[title_col].fillna("").astype(str)
        bodies = df_raw[body_col].fillna("").astype(str)

        df_processed["text_combined"] = titles + " " + bodies

        # Feature Engineering: Metadata
        # Extract specified numerical columns
        for col in Config.METADATA_COLS:
            if col in df_raw.columns:
                # Ensure numeric type
                df_processed[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(
                    0
                )
            else:
                logger.warning(
                    f"Metadata column {col} missing in raw data. Filling with 0."
                )
                df_processed[col] = 0.0

        # 6. Save to Cache
        save_to_cache(df_processed, cache_path)
        logger.info(f"Saved processed {split} data to cache: {cache_path}")

        return df_processed
