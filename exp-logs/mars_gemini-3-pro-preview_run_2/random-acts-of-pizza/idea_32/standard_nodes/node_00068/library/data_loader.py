import os
import json
import pandas as pd
import numpy as np
from library.config import Config


class PizzaDataLoader:
    def __init__(self):
        self.input_dir = Config.INPUT_DIR
        self.metadata_dir = Config.METADATA_DIR
        self.working_dir = Config.WORKING_DIR

        # Ensure working directory exists for caching
        os.makedirs(self.working_dir, exist_ok=True)

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads data for a specific split (train, val, test).
        Merges metadata with raw JSON data.
        Implements caching using Parquet files.
        """
        cache_file = os.path.join(self.working_dir, f"{split}_merged.parquet")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split} data from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
        else:
            # 2. Process from scratch
            print(f"Processing {split} data from scratch...")

            # Load metadata
            meta_path = os.path.join(self.metadata_dir, f"{split}.csv")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df_meta = pd.read_csv(meta_path)

            # Identify necessary raw files from metadata
            # Metadata 'source_file' column contains relative paths like 'input/train.json'
            unique_sources = df_meta["source_file"].unique()

            raw_dfs = []
            for source in unique_sources:
                # Extract filename (e.g., 'train.json') and construct full path
                filename = os.path.basename(source)
                file_path = os.path.join(self.input_dir, filename)

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Raw data file not found: {file_path}")

                with open(file_path, "r") as f:
                    data = json.load(f)

                # Convert to DataFrame
                df_raw_chunk = pd.DataFrame(data)

                # Drop label from raw data if present to avoid conflicts with metadata label
                # Metadata is the source of truth for labels and splits
                if "requester_received_pizza" in df_raw_chunk.columns:
                    df_raw_chunk = df_raw_chunk.drop(
                        columns=["requester_received_pizza"]
                    )

                raw_dfs.append(df_raw_chunk)

            # Concatenate raw data chunks
            if raw_dfs:
                df_raw = pd.concat(raw_dfs, ignore_index=True)
                # Drop duplicates by request_id to ensure clean merge
                df_raw = df_raw.drop_duplicates(subset=["request_id"])

                # Merge: Left join on metadata to preserve split definition
                df = df_meta.merge(df_raw, on="request_id", how="left")
            else:
                df = df_meta

            # Sanitize mixed-type columns before serialization
            if "post_was_edited" in df.columns:
                # Normalize mixed boolean/timestamp values to binary integer
                # Cite debug_lesson_1: Sanitize Mixed-Type Columns Before Parquet Serialization
                df["post_was_edited"] = (
                    df["post_was_edited"].fillna(0).astype(bool).astype(int)
                )

            # Save to cache
            print(f"Saving {split} data to cache: {cache_file}")
            df.to_parquet(cache_file, index=False)

        # 3. Apply Debug Sampling if configured
        if Config.DEBUG_SAMPLE_SIZE is not None:
            print(f"Debug Mode: Sampling first {Config.DEBUG_SAMPLE_SIZE} rows.")
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        return df

    def get_text_corpus(self, df):
        """
        Extracts and concatenates title and text body for embedding.
        Returns a list of strings.
        """
        # Fill NaNs with empty string
        titles = df["request_title"].fillna("").astype(str)

        # Prefer edit_aware text, fallback to standard text if missing (unlikely based on schema)
        if "request_text_edit_aware" in df.columns:
            bodies = df["request_text_edit_aware"].fillna("").astype(str)
        else:
            bodies = df["request_text"].fillna("").astype(str)

        # Concatenate with space
        corpus = (titles + " " + bodies).tolist()
        return corpus

    def get_metadata_features(self, df):
        """
        Extracts numerical metadata features.
        Returns a DataFrame.
        """
        # Define the list of numerical features available at request time
        # We exclude 'at_retrieval' features to prevent data leakage
        feature_cols = [
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

        # Select columns that exist in the dataframe
        available_cols = [c for c in feature_cols if c in df.columns]

        features = df[available_cols].copy()

        # Handle missing values with simple zero filling
        # (Dataset analysis showed no missing values, but this is safe practice)
        features = features.fillna(0)

        return features
