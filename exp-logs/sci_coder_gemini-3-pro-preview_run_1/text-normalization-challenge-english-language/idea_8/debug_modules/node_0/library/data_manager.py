import os
import pandas as pd
import numpy as np
import torch
from library.config import Config


class DataManager:
    def __init__(self, vocab_tokens=None, vocab_chars=None, vocab_classes=None):
        """
        Initializes the DataManager with vocabulary objects.

        Args:
            vocab_tokens: Vocabulary object for tokens (input words).
            vocab_chars: Vocabulary object for characters.
            vocab_classes: Vocabulary object for classes (tags).
        """
        self.vocab_tokens = vocab_tokens
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes

    def load_raw_data(self, split="train"):
        """
        Loads the raw CSV data from the metadata directory.

        Args:
            split (str): One of "train", "val", "test".

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        if split == "train":
            path = Config.TRAIN_DATA_PATH
        elif split == "val":
            path = Config.VAL_DATA_PATH
        elif split == "test":
            path = Config.TEST_DATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Reading raw {split} data from {path}...")
        # keep_default_na=False ensures "NaN" or "null" are read as strings, not NaNs
        df = pd.read_csv(path, dtype=str, keep_default_na=False)

        if Config.DEBUG:
            print(
                f"DEBUG mode: Subsetting {split} data to {Config.DEBUG_SIZE} sentences."
            )
            # Filter by sentence_id to keep sentences intact
            # sentence_id is read as string, assume it can be cast or used as is
            unique_sents = df["sentence_id"].unique()
            if len(unique_sents) > Config.DEBUG_SIZE:
                subset_sents = unique_sents[: Config.DEBUG_SIZE]
                df = df[df["sentence_id"].isin(set(subset_sents))].copy()
                print(f"Subset size: {len(df)} rows")

        return df

    def get_knowledge_base(self, load_cached=True):
        """
        Builds or loads the Knowledge Base (dictionary mapping).

        Args:
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            dict: Mapping {(token, class): normalized_text}
        """
        path = Config.KNOWLEDGE_BASE_PATH

        if load_cached and os.path.exists(path):
            print(f"Loading Knowledge Base from {path}...")
            kb_df = pd.read_parquet(path)
            # Convert to dictionary for O(1) lookup
            kb_dict = {}
            # Iterrows is slow, but acceptable for loading once.
            # Zip is faster.
            for t, c, n in zip(kb_df["token"], kb_df["class"], kb_df["normalized"]):
                kb_dict[(t, c)] = n
            return kb_dict

        print("Building Knowledge Base from training data...")
        df_train = self.load_raw_data("train")

        # We need (before, class) -> after
        # Select relevant columns
        subset = df_train[["before", "class", "after"]].copy()

        # Drop exact duplicates to save space
        subset = subset.drop_duplicates()

        # Handle conflicts: if (before, class) maps to multiple 'after' values.
        # We simply drop duplicates based on keys, keeping the first occurrence.
        # In a real scenario, we might count frequencies, but 'first' is a reasonable deterministic heuristic here.
        subset = subset.drop_duplicates(subset=["before", "class"])

        # Rename for clarity
        kb_df = subset.rename(columns={"before": "token", "after": "normalized"})

        # Save to cache
        print(f"Saving Knowledge Base to {path} ({len(kb_df)} entries)...")
        kb_df.to_parquet(path, index=False)

        # Convert to dict
        kb_dict = {}
        for t, c, n in zip(kb_df["token"], kb_df["class"], kb_df["normalized"]):
            kb_dict[(t, c)] = n

        return kb_dict

    def get_tagger_data(self, split="train", load_cached=True):
        """
        Prepares data for the Bi-LSTM Tagger by grouping tokens into sentences.

        Args:
            split (str): "train", "val", or "test".
            load_cached (bool): Whether to use cached Parquet files.

        Returns:
            pd.DataFrame: DataFrame where each row is a sentence with lists of tokens/classes.
        """
        if split == "train":
            path = Config.TRAIN_GROUPED_PATH
        elif split == "val":
            path = Config.VAL_GROUPED_PATH
        elif split == "test":
            path = Config.TEST_GROUPED_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if load_cached and os.path.exists(path):
            print(f"Loading grouped {split} data from {path}...")
            return pd.read_parquet(path)

        print(f"Processing {split} data for Tagger (Grouping by sentence)...")
        df = self.load_raw_data(split)

        # Ensure correct sorting order: sentence_id, then token_id
        # Convert to int for numerical sorting
        df["sentence_id_int"] = df["sentence_id"].astype(int)
        df["token_id_int"] = df["token_id"].astype(int)
        df = df.sort_values(["sentence_id_int", "token_id_int"])

        # Define aggregation logic
        agg_dict = {"before": list, "id": list}

        # 'class' and 'after' exist only in train/val
        if "class" in df.columns:
            agg_dict["class"] = list
        if "after" in df.columns:
            agg_dict["after"] = list

        # Group by sentence
        grouped = df.groupby("sentence_id_int").agg(agg_dict).reset_index()
        grouped = grouped.rename(columns={"sentence_id_int": "sentence_id"})

        # Save to cache
        print(f"Saving grouped {split} data to {path} ({len(grouped)} sentences)...")
        grouped.to_parquet(path, index=False)

        return grouped

    def get_seq2seq_data(self, split="train", load_cached=True):
        """
        Prepares data for the Seq2Seq Fallback model by filtering for changed tokens.

        Args:
            split (str): "train" or "val".
            load_cached (bool): Whether to use cached Parquet files.

        Returns:
            pd.DataFrame: DataFrame containing only tokens where before != after.
        """
        if split == "train":
            path = Config.TRAIN_SEQ2SEQ_PATH
        elif split == "val":
            path = Config.VAL_SEQ2SEQ_PATH
        else:
            # Test set doesn't have targets, so we can't filter by change
            return None

        if load_cached and os.path.exists(path):
            print(f"Loading seq2seq {split} data from {path}...")
            return pd.read_parquet(path)

        print(f"Processing {split} data for Seq2Seq (Filtering changed tokens)...")
        df = self.load_raw_data(split)

        # Filter where normalization is required
        # Note: df['before'] and df['after'] are strings
        mask = df["before"] != df["after"]
        filtered = df[mask].copy()

        # Keep relevant columns
        filtered = filtered[["before", "class", "after"]]

        # Save to cache
        print(f"Saving seq2seq {split} data to {path} ({len(filtered)} tokens)...")
        filtered.to_parquet(path, index=False)

        return filtered

    def get_class_weights(self, load_cached=True):
        """
        Computes Square-Root Smoothed Class Weights for the Tagger loss function.
        Formula: sqrt(N / N_c) where N is total tokens, N_c is count of class c.

        Args:
            load_cached (bool): Whether to load from .npy file.

        Returns:
            np.ndarray: Array of weights indexed by class ID.
        """
        path = Config.CLASS_WEIGHTS_PATH

        if load_cached and os.path.exists(path):
            print(f"Loading class weights from {path}...")
            return np.load(path)

        print("Computing class weights from training data...")
        if self.vocab_classes is None:
            raise ValueError("vocab_classes must be initialized to compute weights.")

        df = self.load_raw_data("train")

        # Count classes
        class_counts = df["class"].value_counts()
        total_count = len(df)

        # Initialize weights array
        num_classes = len(self.vocab_classes)
        weights = np.zeros(num_classes, dtype=np.float32)

        stoi = self.vocab_classes.get_stoi()

        # Calculate weights
        for cls_name, count in class_counts.items():
            if cls_name in stoi:
                idx = stoi[cls_name]
                # Square-Root Smoothing
                w = np.sqrt(total_count / count)
                weights[idx] = w

        # Handle any classes in vocab that might not be in the loaded split (rare case)
        # Set their weight to 1.0 or mean weight to avoid 0
        if np.any(weights == 0):
            mean_weight = np.mean(weights[weights > 0])
            weights[weights == 0] = mean_weight

        # Save
        print(f"Saving class weights to {path}...")
        np.save(path, weights)

        return weights
