import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class DataLoader:
    """
    Handles loading, merging, and initial processing of the RAOP dataset.
    Implements caching to Parquet files for efficiency.
    """

    def __init__(self):
        self.logger = setup_logger("DataLoader")

        # Define numerical features to extract based on data analysis
        self.numerical_features = [
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

    def _load_raw_json(self, path: str) -> pd.DataFrame:
        """Loads raw JSON data into a DataFrame."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _extract_request_text(self, row: pd.Series) -> str:
        """
        View 1: Semantic Request Content.
        Concatenates title and edit-aware text.
        """
        title = str(row.get("request_title", ""))
        text = str(row.get("request_text_edit_aware", ""))
        return f"{title} {text}".strip()

    def load_dataset(self, split: str, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Loads the dataset for a specific split (train, val, test).

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            pd.DataFrame: The processed DataFrame containing metadata, text views, and labels (if available).
        """
        # Determine paths based on split
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
            cache_path = Config.TRAIN_TABULAR_PATH
            raw_source_key = "train"  # To identify which raw file to load if needed
        elif split == "val":
            meta_path = Config.VAL_META_PATH
            cache_path = Config.VAL_TABULAR_PATH
            raw_source_key = "train"  # Validation comes from train.json
        elif split == "test":
            meta_path = Config.TEST_META_PATH
            cache_path = Config.TEST_TABULAR_PATH
            raw_source_key = "test"
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached {split} data from {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        self.logger.info(f"Processing {split} data from scratch...")

        # Load Metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        df_meta = pd.read_csv(meta_path)

        # Load Raw Data
        # We need to load the specific raw file indicated by the split source
        raw_path = (
            Config.TRAIN_JSON_PATH
            if raw_source_key == "train"
            else Config.TEST_JSON_PATH
        )
        df_raw = self._load_raw_json(raw_path)

        # Merge Metadata with Raw Data
        # We use 'request_id' as the key.
        # Note: df_meta contains the subset of request_ids for this split.
        df_merged = df_meta.merge(df_raw, on="request_id", how="left")

        # Handle potential column name conflicts from merge (though raw shouldn't have label if test)
        # If 'requester_received_pizza' is in both (e.g. train raw and train meta), keep meta
        if "requester_received_pizza_y" in df_merged.columns:
            df_merged = df_merged.drop(columns=["requester_received_pizza_y"])
        if "requester_received_pizza_x" in df_merged.columns:
            df_merged = df_merged.rename(
                columns={"requester_received_pizza_x": "requester_received_pizza"}
            )

        # 3. Feature Extraction

        # View 1: Request Text
        df_merged["text_view"] = df_merged.apply(self._extract_request_text, axis=1)

        # View 3: Numerical Metadata
        # Ensure all expected numerical columns exist, fill missing with 0 or appropriate default
        for col in self.numerical_features:
            if col not in df_merged.columns:
                self.logger.warning(
                    f"Feature {col} missing in raw data. Filling with 0."
                )
                df_merged[col] = 0.0
            else:
                df_merged[col] = df_merged[col].fillna(0.0)

        # Select final columns to keep
        # We keep ID, Label (if exists), Views, and Numericals
        cols_to_keep = ["request_id"]
        if "requester_received_pizza" in df_merged.columns:
            cols_to_keep.append("requester_received_pizza")

        cols_to_keep.extend(["text_view"])
        cols_to_keep.extend(self.numerical_features)

        df_final = df_merged[cols_to_keep]

        # 4. Save to Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_final.to_parquet(cache_path, index=False)
        self.logger.info(f"Saved processed {split} data to {cache_path}")

        return df_final
