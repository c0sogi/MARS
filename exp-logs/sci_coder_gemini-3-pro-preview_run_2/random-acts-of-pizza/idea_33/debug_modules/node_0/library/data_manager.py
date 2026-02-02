import os
import json
import pandas as pd
import numpy as np
from library.utils import setup_logger


class DataManager:
    def __init__(self, cache_dir="./working/idea_33"):
        """
        Initialize the DataManager.

        Args:
            cache_dir (str): Directory to store cached parquet files.
        """
        self.logger = setup_logger("DataManager")
        self.cache_dir = cache_dir
        self.metadata_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "unix_timestamp_of_request",
        ]

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_dataset(self, load_cached_data=True):
        """
        Load the dataset, either from cache or by processing raw files.

        Args:
            load_cached_data (bool): If True, attempt to load from parquet cache.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        train_cache_path = os.path.join(self.cache_dir, "train_processed.parquet")
        val_cache_path = os.path.join(self.cache_dir, "val_processed.parquet")
        test_cache_path = os.path.join(self.cache_dir, "test_processed.parquet")

        if load_cached_data:
            if (
                os.path.exists(train_cache_path)
                and os.path.exists(val_cache_path)
                and os.path.exists(test_cache_path)
            ):
                self.logger.info("Loading data from cache...")
                try:
                    train_df = pd.read_parquet(train_cache_path)
                    val_df = pd.read_parquet(val_cache_path)
                    test_df = pd.read_parquet(test_cache_path)
                    return train_df, val_df, test_df
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}. Reprocessing...")
            else:
                self.logger.info("Cache not found. Processing raw data...")
        else:
            self.logger.info("Ignoring cache. Processing raw data...")

        # Process raw data
        train_df, val_df, test_df = self._process_raw_data()

        # Save to cache
        self.logger.info("Saving processed data to cache...")
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)

        return train_df, val_df, test_df

    def _process_raw_data(self):
        """
        Internal method to read JSONs and Metadata CSVs and construct DataFrames.
        """
        # Load Raw Data
        self.logger.info("Reading raw JSON files...")
        with open("./input/train.json", "r") as f:
            raw_train_data = json.load(f)
        with open("./input/test.json", "r") as f:
            raw_test_data = json.load(f)

        # Load Metadata
        self.logger.info("Reading metadata CSV files...")
        meta_train = pd.read_csv("./metadata/train.csv")
        meta_val = pd.read_csv("./metadata/val.csv")
        meta_test = pd.read_csv("./metadata/test.csv")

        # Helper to extract features from a raw entry
        def extract_features(entry):
            # Text Processing: Title + Body
            title = str(entry.get("request_title", ""))
            body = str(entry.get("request_text_edit_aware", ""))
            if body == "nan" or body is None:
                body = ""
            text_combined = (title + " " + body).strip()

            features = {
                "request_id": entry.get("request_id"),
                "text_combined": text_combined,
            }

            # Metadata Extraction
            for col in self.metadata_cols:
                val = entry.get(col, 0)
                # Ensure numeric
                try:
                    features[col] = float(val)
                except (ValueError, TypeError):
                    features[col] = 0.0

            return features

        # Process Train Split
        self.logger.info("Constructing Train DataFrame...")
        train_records = []
        for _, row in meta_train.iterrows():
            idx = int(row["sample_index"])
            entry = raw_train_data[idx]
            feat = extract_features(entry)
            feat["requester_received_pizza"] = int(row["requester_received_pizza"])
            train_records.append(feat)
        train_df = pd.DataFrame(train_records)

        # Process Val Split
        self.logger.info("Constructing Validation DataFrame...")
        val_records = []
        for _, row in meta_val.iterrows():
            idx = int(row["sample_index"])
            entry = raw_train_data[idx]
            feat = extract_features(entry)
            feat["requester_received_pizza"] = int(row["requester_received_pizza"])
            val_records.append(feat)
        val_df = pd.DataFrame(val_records)

        # Process Test Split
        self.logger.info("Constructing Test DataFrame...")
        test_records = []
        for _, row in meta_test.iterrows():
            idx = int(row["sample_index"])
            entry = raw_test_data[idx]
            feat = extract_features(entry)
            # No label for test
            test_records.append(feat)
        test_df = pd.DataFrame(test_records)

        return train_df, val_df, test_df
