import os
import re
import numpy as np
import pandas as pd
from typing import List, Optional

from library.config import Config, PAD_TOKEN
from library.utils import (
    load_metadata,
    safe_save_dataframe,
    safe_load_dataframe,
    ensure_dir,
    seed_everything,
)


class DataManager:
    """
    Handles data ingestion, sentence reconstruction, and neural dataset preparation.
    """

    def __init__(self, config: Config):
        self.config = config

    def reconstruct_sentences(
        self, split: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Reconstructs full sentences from token-level data.
        Grouping by sentence_id.
        """
        # Define cache path
        filename = f"sentences_{split}_{self.config.get_run_hash()}.parquet"
        cache_path = os.path.join(self.config.working_dir, filename)

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached sentences for {split} from {cache_path}...")
            return safe_load_dataframe(cache_path)

        print(f"Reconstructing sentences for {split}...")

        # 2. Load Raw Data
        df = load_metadata(split, self.config)

        # 3. Group and Aggregate
        # We aggregate relevant columns into lists
        agg_dict = {
            "token_id": list,
            "before": list,
        }

        # Add target columns if they exist (train/val)
        if "after" in df.columns:
            agg_dict["after"] = list
        if "class" in df.columns:
            agg_dict["class"] = list

        # Groupby preserves order of token_id if data is sorted,
        # but to be safe we can sort by token_id within groups if needed.
        # The raw data is usually sorted by sentence_id, token_id.
        # We assume input is sorted.

        df_grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

        # 4. Save to Cache
        print(f"Saving reconstructed sentences to {cache_path}...")
        safe_save_dataframe(df_grouped, cache_path)

        return df_grouped

    def prepare_neural_sequences(
        self, split: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Prepares the token-level dataset for the neural model.
        Extracts context (±2 words) and filters samples based on the strategy.
        """
        # Determine specific cache path based on split
        if split == "train":
            cache_path = self.config.train_seq_path
        elif split == "val":
            cache_path = self.config.val_seq_path
        elif split == "test":
            cache_path = self.config.test_seq_path
        else:
            raise ValueError(f"Unknown split: {split}")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached neural sequences for {split} from {cache_path}...")
            return safe_load_dataframe(cache_path)

        print(f"Preparing neural sequences for {split}...")
        seed_everything(self.config.seed)  # Ensure reproducibility for random sampling

        # 2. Load Raw Data
        df = load_metadata(split, self.config)

        # Ensure sorted by sentence and token for context extraction
        df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

        # 3. Vectorized Context Extraction
        # We use numpy rolling to get neighbors efficiently
        tokens = df["before"].astype(str).values
        sent_ids = df["sentence_id"].values

        # Helper to get shifted array with boundary check
        def get_shift(arr, shift):
            shifted = np.roll(arr, shift)
            # Check boundaries: if sentence_id changed, the context is invalid (from another sentence)
            # For shift +k (future), we compare id[i] vs id[i+k]
            # For shift -k (past), we compare id[i] vs id[i-k]
            # Note: np.roll wraps around, so we must check ids.

            shifted_ids = np.roll(sent_ids, shift)
            mask = sent_ids != shifted_ids

            # Also handle the wrap-around indices explicitly if needed,
            # but the ID check covers it (last sentence ID != first sentence ID usually)
            # Just to be safe for single-sentence datasets, we can rely on ID check.

            shifted[mask] = PAD_TOKEN
            return shifted

        # Context Left
        L1 = get_shift(tokens, 1)  # Previous 1
        L2 = get_shift(tokens, 2)  # Previous 2

        # Context Right
        R1 = get_shift(tokens, -1)  # Next 1
        R2 = get_shift(tokens, -2)  # Next 2

        # Assign to DataFrame
        # We store as lists to match the Tokenizer expectation
        # Using a list comprehension with zip is reasonably fast for combining
        # or we can just save columns and combine in Dataset.
        # The prompt requirement implies "create_neural_split" -> "Construct a training set".
        # Let's create the columns 'context_left' and 'context_right' as lists of strings.

        print("Constructing context columns...")
        # Zip is faster than row iteration
        df["context_left"] = [list(x) for x in zip(L2, L1)]
        df["context_right"] = [list(x) for x in zip(R1, R2)]

        # 4. Filter Data
        print("Filtering data...")
        if split in ["train", "val"]:
            # Logic:
            # 1. Keep all semiotic classes (NOT PLAIN/PUNCT)
            # 2. Keep all tokens with digits (even if PLAIN, though rare)
            # 3. Keep 30% of PLAIN/PUNCT

            # Identify classes
            is_plain_punct = df["class"].isin(["PLAIN", "PUNCT"])

            # Identify digits
            has_digit = df["before"].str.contains(r"\d", regex=True)

            # Random mask for background
            random_mask = np.random.rand(len(df)) < 0.3

            # Combine
            keep_mask = (~is_plain_punct) | (has_digit) | (random_mask)

            df_filtered = df[keep_mask].copy()

        elif split == "test":
            # Logic:
            # The Neural Router only activates if token contains digits.
            # So we only need to predict for tokens with digits.
            # The rest are handled by N-grams.

            has_digit = df["before"].str.contains(r"\d", regex=True)
            df_filtered = df[has_digit].copy()

            # Note: We must preserve 'id' (sentence_id + token_id) to map back predictions.
            # The 'id' column is already constructed in the raw data?
            # Metadata 'test.csv' has sentence_id, token_id.
            # We should construct the submission 'id' here for convenience.
            df_filtered["id"] = (
                df_filtered["sentence_id"].astype(str)
                + "_"
                + df_filtered["token_id"].astype(str)
            )

        else:
            df_filtered = df.copy()

        # 5. Select Columns
        cols = ["sentence_id", "token_id", "before", "context_left", "context_right"]
        if "after" in df_filtered.columns:
            cols.append("after")
        if "class" in df_filtered.columns:
            cols.append("class")
        if "id" in df_filtered.columns:
            cols.append("id")

        df_final = df_filtered[cols]

        print(f"Original samples: {len(df)}")
        print(f"Filtered samples: {len(df_final)}")

        # 6. Save to Cache
        print(f"Saving neural sequences to {cache_path}...")
        safe_save_dataframe(df_final, cache_path)

        return df_final
