import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


class DataLoader:
    """
    Handles ingestion, merging, and basic preprocessing of the Random Acts of Pizza dataset.
    Implements caching to speed up iterative development.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        set_seed(Config.SEED)

    def load_merged_data(self, split="train", load_cached_data=True):
        """
        Loads the dataset for a specific split (train, val, test), merges it with metadata,
        performs basic cleaning, and caches the result.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from local cache first.

        Returns:
            pd.DataFrame: The processed dataframe for the requested split.
        """
        cache_path = os.path.join(self.working_dir, f"{split}_merged.parquet")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Processing {split} data from scratch...")

        # 2. Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
        elif split == "val":
            meta_path = Config.VAL_META_PATH
        elif split == "test":
            meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # 3. Load Raw JSON Data
        # Determine which raw file to load based on the split
        # train and val come from train.json, test comes from test.json
        if split in ["train", "val"]:
            raw_path = Config.TRAIN_DATA_PATH
        else:
            raw_path = Config.TEST_DATA_PATH

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        with open(raw_path, "r") as f:
            raw_data = json.load(f)

        df_raw = pd.DataFrame(raw_data)

        # 4. Merge Metadata with Raw Data
        # We use inner join on request_id to keep only the samples belonging to this split
        # The metadata contains the ground truth label for train/val
        df_merged = df_meta.merge(df_raw, on="request_id", how="inner")

        # Handle column name conflicts if any (e.g., if label is in both)
        # Metadata source of truth for labels is preferred usually, but they should match.
        if "requester_received_pizza_x" in df_merged.columns:
            df_merged.rename(
                columns={"requester_received_pizza_x": "requester_received_pizza"},
                inplace=True,
            )
            df_merged.drop(
                columns=["requester_received_pizza_y"], inplace=True, errors="ignore"
            )

        # 5. Preprocessing
        df_processed = self.prepare_text_fields(df_merged)
        df_processed = self.clean_metadata(df_processed)

        # Debugging: Subsample if configured
        if Config.DEBUG and Config.MAX_SAMPLES:
            print(f"DEBUG MODE: Subsampling to {Config.MAX_SAMPLES} samples.")
            df_processed = df_processed.head(Config.MAX_SAMPLES)

        # 6. Save to Cache
        # Parquet handles lists (like subreddits) reasonably well with pyarrow engine
        try:
            df_processed.to_parquet(cache_path, index=False)
            print(f"Saved {split} data to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

        return df_processed

    def prepare_text_fields(self, df):
        """
        Concatenates title and body text, handling missing values.
        """
        df = df.copy()

        # Fill NaNs
        df["request_title"] = df["request_title"].fillna("")

        # Use edit_aware text if available, else standard text, else empty
        text_col = (
            "request_text_edit_aware"
            if "request_text_edit_aware" in df.columns
            else "request_text"
        )
        if text_col not in df.columns:
            # Fallback if neither exists (unlikely given dataset spec)
            df["text_body"] = ""
        else:
            df[text_col] = df[text_col].fillna("")

        # Create combined text column for embeddings
        # Format: "Title. Body"
        df["text_combined"] = (
            df["request_title"].astype(str) + " " + df[text_col].astype(str)
        )

        return df

    def clean_metadata(self, df):
        """
        Cleans metadata columns: handles missing values and ensures correct types.
        """
        df = df.copy()

        # Ensure numerical columns are float/int and fill NaNs
        # Based on data analysis, there were no missing values in train, but we safeguard for test
        numeric_cols = [
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

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Ensure 'requester_subreddits_at_request' is a list of strings
        # JSON loading usually produces lists, but we ensure it's not None
        if "requester_subreddits_at_request" in df.columns:
            df["requester_subreddits_at_request"] = df[
                "requester_subreddits_at_request"
            ].apply(lambda x: x if isinstance(x, list) else [])

        return df
