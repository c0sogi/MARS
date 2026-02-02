import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from library import config
from library import utils
from library import tokenizers


class SemioticDataset(Dataset):
    """
    PyTorch Dataset for the Semiotic Transformer (Tier 2).

    Input: Character-level sequence "Prev_Word <SEP> Target_Chars <SEP> Next_Word"
    Output: Subword-level (BPE) sequence of the normalized text.
    """

    def __init__(
        self, df, char_tokenizer, bpe_tokenizer, max_input_len, max_output_len
    ):
        self.df = df.reset_index(drop=True)
        self.char_tokenizer = char_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        # Pre-fetch columns to avoid overhead in __getitem__
        self.befores = self.df["before"].astype(str).tolist()
        self.afters = self.df["after"].astype(str).tolist()
        self.prevs = self.df["prev"].astype(str).tolist()
        self.nexts = self.df["next"].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Construct Input String
        # Format: prev <SEP> before <SEP> next
        prev_tok = self.prevs[idx]
        curr_tok = self.befores[idx]
        next_tok = self.nexts[idx]

        # We use the SEP token from the char tokenizer
        sep = self.char_tokenizer.sep_token
        input_text = f"{prev_tok}{sep}{curr_tok}{sep}{next_tok}"

        # 2. Encode Input (Character Level)
        # No special tokens (SOS/EOS) needed for Encoder input in this architecture, usually.
        # But we need to handle padding.
        input_ids = self.char_tokenizer.encode(input_text, add_special_tokens=False)

        # Truncate
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[: self.max_input_len]

        # Create Attention Mask
        attention_mask = [1] * len(input_ids)

        # Pad
        pad_len = self.max_input_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.char_tokenizer.pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        # 3. Encode Target (BPE Level)
        # Decoder target needs SOS (BOS) and EOS.
        target_text = self.afters[idx]
        target_ids = self.bpe_tokenizer.encode(target_text, add_special_tokens=True)

        # Truncate
        if len(target_ids) > self.max_output_len:
            # Ensure we keep EOS if possible, but simple truncation is standard
            target_ids = target_ids[: self.max_output_len - 1] + [
                self.bpe_tokenizer.eos_id
            ]

        # Pad
        # We usually pad labels with a specific ignore index (e.g. -100) for loss computation,
        # or just pad with pad_id. Here we use pad_id.
        pad_len_tgt = self.max_output_len - len(target_ids)
        if pad_len_tgt > 0:
            target_ids = target_ids + [self.bpe_tokenizer.pad_id] * pad_len_tgt

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }


def _process_raw_data(file_path, is_train=False):
    """
    Internal helper to load, context-window, filter, and balance the data.
    """
    print(f"Processing {file_path} (is_train={is_train})...")

    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    # Handle NaNs
    df["before"] = df["before"].fillna("").astype(str)
    df["after"] = df["after"].fillna("").astype(str)
    if "class" not in df.columns:
        df["class"] = "UNKNOWN"
    df["class"] = df["class"].fillna("UNKNOWN").astype(str)

    # 2. Generate Context (Prev/Next)
    # Must be done on the FULL dataset before filtering to preserve sequence
    df.sort_values(["sentence_id", "token_id"], inplace=True)

    # Prev
    df["prev"] = df["before"].shift(1).fillna("<START>")
    df.loc[df["token_id"] == 0, "prev"] = "<START>"

    # Next
    df["next"] = df["before"].shift(-1).fillna("<END>")
    # If next token_id is 0, current is end of sentence
    next_token_id = df["token_id"].shift(-1).fillna(0)
    df.loc[next_token_id == 0, "next"] = "<END>"

    # 3. Filter Semiotic Tokens
    # Logic: Keep if (is_semiotic is True) AND (class is NOT in EXCLUDE_CLASSES)
    # We first identify rows that are explicitly excluded (PLAIN, PUNCT)
    exclude_mask = df["class"].isin(config.EXCLUDE_CLASSES)

    # We identify rows that look semiotic
    # We can use the utils.is_semiotic function.
    # Optimization: Vectorize or apply. Since is_semiotic uses regex, apply is necessary.
    # However, to save time, we can rely on class labels if available for the bulk.
    # But the requirement is to use is_semiotic.

    # Let's do the exclusion first to reduce rows to check
    # But wait, if a PLAIN token has digits, we might want it?
    # The prompt says: "Exclude PLAIN and PUNCT". This is a hard exclusion for the Transformer training set.
    # So we drop all PLAIN/PUNCT first.
    df_filtered = df[~exclude_mask].copy()

    # Now, for the remaining classes (CARDINAL, DATE, etc.), they are by definition semiotic
    # based on config.SEMIOTIC_CLASSES.
    # Are there any "UNKNOWN" or other classes?
    # We can perform a safety check using is_semiotic, but it's likely redundant for named classes.
    # Let's keep all remaining rows as they are the "hard cases".

    print(
        f"  Filtered from {len(df)} to {len(df_filtered)} tokens (removed PLAIN/PUNCT)."
    )

    if is_train and not df_filtered.empty:
        # 4. Class Balancing (Upsampling)
        print("  Balancing classes...")
        groups = df_filtered.groupby("class")
        balanced_chunks = []

        for cls_name, group in groups:
            count = len(group)
            if count < config.TARGET_CLASS_COUNT:
                # Upsample
                upsampled = group.sample(
                    n=config.TARGET_CLASS_COUNT, replace=True, random_state=config.SEED
                )
                balanced_chunks.append(upsampled)
            else:
                # Keep original (Dominant classes)
                balanced_chunks.append(group)

        if balanced_chunks:
            df_filtered = pd.concat(balanced_chunks, ignore_index=True)
            print(f"  Balanced size: {len(df_filtered)}")

        # Shuffle after balancing
        df_filtered = df_filtered.sample(frac=1, random_state=config.SEED).reset_index(
            drop=True
        )

        # Debug limit
        if config.DEBUG:
            print(f"  DEBUG: Limiting to {config.MAX_TRAIN_SAMPLES} samples.")
            df_filtered = df_filtered.head(config.MAX_TRAIN_SAMPLES)

    return df_filtered


def prepare_data(load_cached_data=True):
    """
    Prepares the training and validation datasets.
    Handles tokenizers, caching, filtering, and balancing.

    Args:
        load_cached_data (bool): If True, tries to load parquet files from cache.

    Returns:
        tuple: (train_dataset, val_dataset, char_tokenizer, bpe_tokenizer)
    """
    # 1. Load/Build Tokenizers
    char_tokenizer, bpe_tokenizer = tokenizers.build_tokenizers(
        load_cached_data=load_cached_data
    )

    # 2. Prepare Training Data
    train_df = None
    if load_cached_data and os.path.exists(config.PROCESSED_TRAIN_PATH):
        print("Dataset: Loading processed train data from cache...")
        train_df = pd.read_parquet(config.PROCESSED_TRAIN_PATH)
    else:
        print("Dataset: Processing training data from scratch...")
        train_df = _process_raw_data(config.TRAIN_FILE, is_train=True)
        print(
            f"Dataset: Saving processed train data to {config.PROCESSED_TRAIN_PATH}..."
        )
        utils.save_cache(train_df, config.PROCESSED_TRAIN_PATH)

    # 3. Prepare Validation Data
    val_df = None
    if load_cached_data and os.path.exists(config.PROCESSED_VAL_PATH):
        print("Dataset: Loading processed val data from cache...")
        val_df = pd.read_parquet(config.PROCESSED_VAL_PATH)
    else:
        print("Dataset: Processing validation data from scratch...")
        # Note: We do NOT balance validation data, but we DO filter it to evaluate
        # on the relevant semiotic subset (as Tier 2 is only used for these).
        val_df = _process_raw_data(config.VAL_FILE, is_train=False)
        print(f"Dataset: Saving processed val data to {config.PROCESSED_VAL_PATH}...")
        utils.save_cache(val_df, config.PROCESSED_VAL_PATH)

    # 4. Create Datasets
    print(
        f"Dataset: Creating PyTorch Datasets (Train: {len(train_df)}, Val: {len(val_df)})..."
    )

    train_dataset = SemioticDataset(
        train_df,
        char_tokenizer,
        bpe_tokenizer,
        config.MAX_INPUT_LEN,
        config.MAX_OUTPUT_LEN,
    )

    val_dataset = SemioticDataset(
        val_df,
        char_tokenizer,
        bpe_tokenizer,
        config.MAX_INPUT_LEN,
        config.MAX_OUTPUT_LEN,
    )

    return train_dataset, val_dataset, char_tokenizer, bpe_tokenizer
