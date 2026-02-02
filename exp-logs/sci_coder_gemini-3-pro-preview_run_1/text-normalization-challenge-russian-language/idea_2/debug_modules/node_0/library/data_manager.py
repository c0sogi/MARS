import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from library.config import Config
from library.tokenizer import CharTokenizer


def build_ngram_stats(train_path, load_cached=True):
    """
    Builds or loads hierarchical N-gram statistics (Trigram, Bigram, Unigram)
    from the training data.

    Args:
        train_path (str): Path to the training CSV file.
        load_cached (bool): Whether to attempt loading from the cache.

    Returns:
        dict: A dictionary containing 'unigram', 'bigram', and 'trigram' lookups.
    """
    cache_path = Config.NGRAM_STATS_PATH

    # 1. Try to load from cache
    if load_cached and os.path.exists(cache_path):
        print(f"Loading N-gram stats from {cache_path}...")
        try:
            # Allow pickle is required for loading dictionaries with object keys/values
            stats = np.load(cache_path, allow_pickle=True).item()
            return stats
        except Exception as e:
            print(f"Failed to load cached stats: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Building N-gram stats from scratch...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load data
    # We need sentence_id to ensure N-grams don't cross sentence boundaries
    usecols = ["sentence_id", "token_id", "before", "after"]
    try:
        df = pd.read_csv(train_path, usecols=usecols, dtype=str)
    except ValueError:
        # Fallback if columns are missing or named differently
        df = pd.read_csv(train_path, dtype=str)

    # Fill NaNs
    df["before"] = df["before"].fillna("")
    df["after"] = df["after"].fillna("")

    # Ensure IDs are integers for sorting
    df["sentence_id"] = df["sentence_id"].astype(int)
    df["token_id"] = df["token_id"].astype(int)

    # Sort to ensure correct order
    df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

    # Create shifted columns for context
    df["prev"] = df["before"].shift(1).fillna("")
    df["prev2"] = df["before"].shift(2).fillna("")
    df["prev_sent"] = df["sentence_id"].shift(1).fillna(-1)
    df["prev2_sent"] = df["sentence_id"].shift(2).fillna(-1)

    # Identify valid contexts (must be within same sentence)
    valid_bigram = df["sentence_id"] == df["prev_sent"]
    valid_trigram = (df["sentence_id"] == df["prev_sent"]) & (
        df["sentence_id"] == df["prev2_sent"]
    )

    # --- Unigram Stats ---
    # Map: before -> after (most freq)
    print("Computing Unigram Stats...")
    unigram_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
    # Sort by count desc, then drop duplicates to keep top
    best_unigrams = unigram_counts.sort_values(
        "count", ascending=False
    ).drop_duplicates("before")
    unigram_dict = dict(zip(best_unigrams["before"], best_unigrams["after"]))

    # --- Bigram Stats ---
    # Map: (prev, before) -> after
    print("Computing Bigram Stats...")
    df_bi = df[valid_bigram].copy()
    bigram_counts = (
        df_bi.groupby(["prev", "before", "after"]).size().reset_index(name="count")
    )
    best_bigrams = bigram_counts.sort_values("count", ascending=False).drop_duplicates(
        ["prev", "before"]
    )
    # Fast dict creation
    bigram_keys = list(zip(best_bigrams["prev"], best_bigrams["before"]))
    bigram_dict = dict(zip(bigram_keys, best_bigrams["after"]))

    # --- Trigram Stats ---
    # Map: (prev2, prev, before) -> after
    print("Computing Trigram Stats...")
    df_tri = df[valid_trigram].copy()
    trigram_counts = (
        df_tri.groupby(["prev2", "prev", "before", "after"])
        .size()
        .reset_index(name="count")
    )
    best_trigrams = trigram_counts.sort_values(
        "count", ascending=False
    ).drop_duplicates(["prev2", "prev", "before"])
    # Fast dict creation
    trigram_keys = list(
        zip(best_trigrams["prev2"], best_trigrams["prev"], best_trigrams["before"])
    )
    trigram_dict = dict(zip(trigram_keys, best_trigrams["after"]))

    stats = {"unigram": unigram_dict, "bigram": bigram_dict, "trigram": trigram_dict}

    # Save
    print(f"Saving N-gram stats to {cache_path}...")
    np.save(cache_path, stats)

    return stats


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for the Neural Normalization Model.
    Filters data to focus on complex tokens (digits) and provides context.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=128,
        context_window=1,
        mode="train",
        load_cached=True,
    ):
        """
        Args:
            data_path (str): Path to csv file (train or val).
            tokenizer (CharTokenizer): Fitted tokenizer instance.
            max_len (int): Max sequence length for tokenizer.
            context_window (int): Number of tokens to use as context on each side.
            mode (str): 'train' or 'val'.
            load_cached (bool): Unused argument kept for API consistency.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.context_window = context_window
        self.mode = mode

        # Load Data
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, dtype=str)

        # Handle NaNs
        df["before"] = df["before"].fillna("")
        if "after" in df.columns:
            df["after"] = df["after"].fillna("")
        else:
            df["after"] = (
                ""  # For test set if needed, though this dataset is mostly for training
            )

        # Convert IDs
        df["sentence_id"] = df["sentence_id"].astype(int)
        df["token_id"] = df["token_id"].astype(int)

        # Sort to ensure correct order for context extraction
        df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

        self.data = df

        # Filter indices for training/validation
        # We only want to train on tokens that the Router would send to this model.
        # Router Condition: Contains a digit (Config.DIGIT_REGEX)

        print(f"Filtering {mode} dataset for digit-containing tokens...")
        digit_mask = df["before"].str.contains(Config.DIGIT_REGEX, regex=True)

        # Get indices of interesting tokens
        self.indices = df.index[digit_mask].tolist()

        # Debug Subset
        if Config.DEBUG_SUBSET_SIZE and mode == "train":
            print(f"Subsetting to {Config.DEBUG_SUBSET_SIZE} samples for debugging...")
            if len(self.indices) > Config.DEBUG_SUBSET_SIZE:
                # Deterministic subset
                self.indices = self.indices[: Config.DEBUG_SUBSET_SIZE]

        print(
            f"Initialized {mode} dataset with {len(self.indices)} samples (from {len(df)} total tokens)."
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the global index of the target token
        global_idx = self.indices[idx]

        # Get the target row
        row = self.data.iloc[global_idx]

        curr_token = row["before"]
        target_token = row["after"]
        sentence_id = row["sentence_id"]

        # Construct Context
        # We look at rows around global_idx

        # Left Context
        left_tokens = []
        for i in range(1, self.context_window + 1):
            prev_idx = global_idx - i
            if prev_idx >= 0:
                # Accessing single row via iloc is relatively fast for batch sizes
                prev_row = self.data.iloc[prev_idx]
                if prev_row["sentence_id"] == sentence_id:
                    left_tokens.insert(0, prev_row["before"])
                else:
                    break  # Sentence boundary
            else:
                break

        # Right Context
        right_tokens = []
        for i in range(1, self.context_window + 1):
            next_idx = global_idx + i
            if next_idx < len(self.data):
                next_row = self.data.iloc[next_idx]
                if next_row["sentence_id"] == sentence_id:
                    right_tokens.append(next_row["before"])
                else:
                    break
            else:
                break

        # Combine
        # Format: "prev curr next"
        # We join with spaces. The tokenizer is char-level, so spaces are treated as characters.
        input_str = " ".join(left_tokens + [curr_token] + right_tokens)
        target_str = target_token

        # Tokenize
        src_enc = self.tokenizer.encode(
            input_str, max_len=self.max_len, add_special_tokens=True, return_tensor=True
        )
        tgt_enc = self.tokenizer.encode(
            target_str,
            max_len=self.max_len,
            add_special_tokens=True,
            return_tensor=True,
        )

        return {
            "src": src_enc,
            "tgt": tgt_enc,
            "original_before": curr_token,
            "original_after": target_token,
        }
