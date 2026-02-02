import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class DataLoader:
    """
    DataLoader class handles the ingestion of raw JSON data and metadata,
    merging them into structured DataFrames, and extracting relevant features.
    """

    def __init__(self):
        self.logger = setup_logger("DataLoader")
        # Ensure working directory exists as per requirements
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def load_data(self, load_cached: bool = True):
        """
        Loads training, validation, and test datasets.

        Args:
            load_cached (bool): If True, attempts to load from parquet cache.
                                If False or cache missing, reprocesses from raw data.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        self.logger.info("Starting data loading process...")

        train_df = self._load_split(
            split_name="train",
            meta_path=Config.TRAIN_META_PATH,
            raw_path=Config.TRAIN_JSON_PATH,
            cache_path=Config.TRAIN_FEATURES_PATH,
            load_cached=load_cached,
        )

        val_df = self._load_split(
            split_name="val",
            meta_path=Config.VAL_META_PATH,
            raw_path=Config.TRAIN_JSON_PATH,
            cache_path=Config.VAL_FEATURES_PATH,
            load_cached=load_cached,
        )

        test_df = self._load_split(
            split_name="test",
            meta_path=Config.TEST_META_PATH,
            raw_path=Config.TEST_JSON_PATH,
            cache_path=Config.TEST_FEATURES_PATH,
            load_cached=load_cached,
        )

        self.logger.info("Data loading complete.")
        return train_df, val_df, test_df

    def _load_split(self, split_name, meta_path, raw_path, cache_path, load_cached):
        """
        Internal method to load a specific data split.
        """
        # 1. Try Cache
        if load_cached and os.path.exists(cache_path):
            self.logger.info(f"Loading cached {split_name} features from {cache_path}")
            return pd.read_parquet(cache_path)

        self.logger.info(f"Processing {split_name} data from scratch...")

        # 2. Load Metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        meta_df = pd.read_csv(meta_path)

        # Handle Debugging
        if Config.DEBUG:
            self.logger.info(
                f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows for {split_name}."
            )
            meta_df = meta_df.head(Config.DEBUG_SAMPLE_SIZE)

        # 3. Load Raw JSON
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        with open(raw_path, "r") as f:
            raw_data = json.load(f)

        # 4. Merge and Extract Features
        processed_records = []

        # Metadata contains 'sample_index' which corresponds to the index in the raw JSON list
        for _, row in meta_df.iterrows():
            idx = int(row["sample_index"])

            # Safety check for index bounds
            if idx >= len(raw_data):
                self.logger.warning(
                    f"Sample index {idx} out of bounds for {split_name}. Skipping."
                )
                continue

            raw_entry = raw_data[idx]

            # Extract features
            record = self._extract_features(raw_entry)

            # Add Identifiers
            record["request_id"] = row["request_id"]

            # Add Target if present in metadata (train/val)
            if "requester_received_pizza" in row:
                record["requester_received_pizza"] = int(
                    row["requester_received_pizza"]
                )

            processed_records.append(record)

        df = pd.DataFrame(processed_records)

        # 5. Save to Cache
        self.logger.info(f"Saving {split_name} features to {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df

    def _extract_features(self, entry):
        """
        Extracts text and numerical metadata from a single raw JSON entry.
        """
        features = {}

        # --- Text Data ---
        # Use 'request_text_edit_aware' to avoid leakage from edits saying "Thanks for pizza"
        title = entry.get("request_title", "")
        body = entry.get("request_text_edit_aware", "")

        # Handle None/Null
        if title is None:
            title = ""
        if body is None:
            body = ""

        features["request_title"] = str(title)
        features["request_text_edit_aware"] = str(body)
        # Concatenation for Global Context View
        features["text_concat"] = str(title) + " " + str(body)

        # --- Numerical Metadata ---
        # List of numerical features to extract based on analysis and idea
        num_cols = [
            "unix_timestamp_of_request",
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        for col in num_cols:
            val = entry.get(col, 0.0)
            # Simple imputation for missing values in raw json
            if val is None:
                val = 0.0
            features[col] = float(val)

        return features
