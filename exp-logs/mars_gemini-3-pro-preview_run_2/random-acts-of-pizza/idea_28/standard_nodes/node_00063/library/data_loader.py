import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, set_seed

logger = setup_logger("data_loader")


class DataLoader:
    """
    Handles loading of raw JSON data and metadata CSVs, processing them into
    structured DataFrames with text and numerical features, and caching the results.
    """

    def __init__(self):
        self.config = Config
        # Define validation cache path (not explicitly in Config but required for split)
        self.val_metadata_path = os.path.join(
            self.config.WORKING_DIR, "val_metadata.parquet"
        )

    def load_raw_json(self, path):
        """Loads raw JSON data from the specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def process_split(self, meta_df, raw_data, is_test=False):
        """
        Extracts features from raw data based on metadata indices.

        Args:
            meta_df (pd.DataFrame): Metadata containing sample indices.
            raw_data (list): List of dictionaries from the raw JSON.
            is_test (bool): Whether processing the test set (no labels).

        Returns:
            pd.DataFrame: Processed dataframe with text, metadata, and labels.
        """
        records = []

        # Iterate through metadata to pull correct records from raw JSON list
        for _, row in meta_df.iterrows():
            idx = int(row["sample_index"])

            if idx < 0 or idx >= len(raw_data):
                logger.warning(f"Sample index {idx} out of bounds. Skipping.")
                continue

            entry = raw_data[idx]

            record = {}
            record["request_id"] = entry.get("request_id")

            # Extract Label (from metadata for reliability)
            if not is_test:
                record["requester_received_pizza"] = int(
                    row["requester_received_pizza"]
                )

            # Extract and Concatenate Text Features
            text_parts = []
            for col in self.config.TEXT_COLS:
                val = entry.get(col, "")
                if val is None:
                    val = ""
                text_parts.append(str(val))
            record["text_combined"] = " ".join(text_parts).strip()

            # Extract Numerical Metadata Features
            for col in self.config.METADATA_COLS:
                val = entry.get(col, np.nan)
                record[col] = val

            records.append(record)

        return pd.DataFrame(records)

    def load_data(self, load_cached_data=True):
        """
        Main method to load data. Checks cache first, otherwise processes from raw.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        set_seed()

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Check if cache files exist
        cache_exists = (
            os.path.exists(self.config.TRAIN_METADATA_PATH)
            and os.path.exists(self.val_metadata_path)
            and os.path.exists(self.config.TEST_METADATA_PATH)
        )

        if load_cached_data and cache_exists:
            logger.info("Loading data from parquet cache...")
            try:
                train_df = pd.read_parquet(self.config.TRAIN_METADATA_PATH)
                val_df = pd.read_parquet(self.val_metadata_path)
                test_df = pd.read_parquet(self.config.TEST_METADATA_PATH)
                logger.info(
                    f"Loaded train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
                )
                return train_df, val_df, test_df
            except Exception as e:
                logger.error(f"Failed to load cache: {e}. Reprocessing from scratch.")

        logger.info("Processing data from raw sources...")

        # Load Raw JSON
        logger.info(f"Loading raw JSON from {self.config.TRAIN_JSON_PATH}")
        train_raw_data = self.load_raw_json(self.config.TRAIN_JSON_PATH)

        logger.info(f"Loading raw JSON from {self.config.TEST_JSON_PATH}")
        test_raw_data = self.load_raw_json(self.config.TEST_JSON_PATH)

        # Load Metadata CSVs
        logger.info("Loading metadata splits...")
        train_meta = pd.read_csv(self.config.TRAIN_META_PATH)
        val_meta = pd.read_csv(self.config.VAL_META_PATH)
        test_meta = pd.read_csv(self.config.TEST_META_PATH)

        # Process Splits
        # Note: Train and Val splits both reference the training JSON file
        logger.info("Processing Train split...")
        train_df = self.process_split(train_meta, train_raw_data, is_test=False)

        logger.info("Processing Validation split...")
        val_df = self.process_split(val_meta, train_raw_data, is_test=False)

        logger.info("Processing Test split...")
        test_df = self.process_split(test_meta, test_raw_data, is_test=True)

        # Save to Cache
        logger.info("Saving processed data to cache...")
        try:
            train_df.to_parquet(self.config.TRAIN_METADATA_PATH, index=False)
            val_df.to_parquet(self.val_metadata_path, index=False)
            test_df.to_parquet(self.config.TEST_METADATA_PATH, index=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        logger.info(
            f"Processed train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
        )
        return train_df, val_df, test_df


def load_data(load_cached_data=True):
    """Wrapper function to instantiate DataLoader and load data."""
    loader = DataLoader()
    return loader.load_data(load_cached_data=load_cached_data)
