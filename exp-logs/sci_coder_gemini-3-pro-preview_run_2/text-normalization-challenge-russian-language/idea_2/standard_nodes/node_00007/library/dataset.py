import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.utils import save_data, load_data, ensure_dir
from library.vocab import CharTokenizer


def process_data(mode, load_cached_data=True):
    """
    Loads, processes, and caches the dataset for the Seq2Seq model.

    Processing steps:
    1. Load raw CSV based on mode (train/val/test).
    2. Sort by sentence_id and token_id to ensure correct order.
    3. Generate context columns (prev_before, next_before) handling sentence boundaries.
    4. Filter rows to keep only tokens containing digits (0-9).
    5. Cache the processed DataFrame to Parquet.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    cache_filename = f"processed_{mode}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading processed {mode} data from cache: {cache_path}")
            return load_data(cache_path)
        except Exception:
            pass

    # 2. Compute from scratch
    # Determine source file
    if mode == "train":
        file_path = Config.TRAIN_DATA_PATH
    elif mode == "val":
        file_path = Config.VAL_DATA_PATH
    elif mode == "test":
        file_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # print(f"Processing {mode} data from {file_path}...")
    df = pd.read_csv(file_path)

    # Ensure string types
    df["before"] = df["before"].fillna("").astype(str)
    if "after" in df.columns:
        df["after"] = df["after"].fillna("").astype(str)

    # Sort to ensure context logic works
    # We assume sentence_id and token_id are present
    if "sentence_id" in df.columns and "token_id" in df.columns:
        df.sort_values(["sentence_id", "token_id"], inplace=True)

    # Generate Context
    # Shift to get previous and next tokens
    df["prev_before"] = df["before"].shift(1).fillna("<start>")
    df["next_before"] = df["before"].shift(-1).fillna("<end>")

    # Handle Sentence Boundaries
    # If sentence_id exists, use it to break context
    if "sentence_id" in df.columns:
        # If prev sentence != current sentence, prev_token is <start>
        mask_start = df["sentence_id"] != df["sentence_id"].shift(1)
        df.loc[mask_start, "prev_before"] = "<start>"

        # If next sentence != current sentence, next_token is <end>
        mask_end = df["sentence_id"] != df["sentence_id"].shift(-1)
        df.loc[mask_end, "next_before"] = "<end>"

    # Filter for Digits
    # We only want to train the NN on tokens that contain digits
    # Regex checks for any digit 0-9
    digit_mask = df["before"].str.contains(r"\d", regex=True)
    df_filtered = df[digit_mask].copy()

    # 3. Save to cache
    save_data(df_filtered, cache_path)

    return df_filtered


class DigitSeq2SeqDataset(Dataset):
    """
    PyTorch Dataset for Character-Level Seq2Seq Text Normalization.
    Filters data to only include tokens with digits.
    Constructs input as: [Prev] <SEP> [Target] <SEP> [Next]
    """

    def __init__(
        self, mode="train", tokenizer=None, load_cached_data=True, debug=False
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            tokenizer (CharTokenizer): Initialized tokenizer instance.
            load_cached_data (bool): Whether to use cached processed data.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.mode = mode
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to DigitSeq2SeqDataset.")

        # Load and process data
        self.data = process_data(mode, load_cached_data=load_cached_data)

        # Debug mode: slice data
        if debug:
            self.data = self.data.iloc[: Config.DEBUG_SIZE].copy()

        # Reset index to ensure standard integer indexing for __getitem__
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Construct Source Sequence
        # Format: prev <sep> current <sep> next
        prev_token = str(row["prev_before"])
        curr_token = str(row["before"])
        next_token = str(row["next_before"])

        sep = Config.SEP_TOKEN
        source_text = f"{prev_token}{sep}{curr_token}{sep}{next_token}"

        # Encode Source
        # We use MAX_SEQ_LEN from config
        src_indices = self.tokenizer.encode(
            source_text,
            max_len=Config.MAX_SEQ_LEN,
            add_special_tokens=True,  # Adds SOS/EOS
        )

        result = {
            "src": src_indices,
            "raw_before": curr_token,
            "raw_prev": prev_token,
            "raw_next": next_token,
        }

        # Handle ID for submission
        if "sentence_id" in row and "token_id" in row:
            result["id"] = f"{row['sentence_id']}_{row['token_id']}"
        else:
            # Fallback if columns missing (unlikely given process_data logic)
            result["id"] = str(idx)

        # Handle Target (only for train/val)
        if self.mode in ["train", "val"]:
            target_text = str(row["after"])
            tgt_indices = self.tokenizer.encode(
                target_text, max_len=Config.MAX_SEQ_LEN, add_special_tokens=True
            )
            result["tgt"] = tgt_indices
            result["raw_after"] = target_text

        return result

    def collate_fn(self, batch):
        """
        Custom collate function to pad sequences.
        """
        src_list = [item["src"] for item in batch]

        # Pad source sequences
        # batch_first=True -> (Batch, Seq)
        src_padded = pad_sequence(
            src_list, batch_first=True, padding_value=Config.PAD_IDX
        )

        batch_data = {
            "src": src_padded,
            "raw_before": [item["raw_before"] for item in batch],
            "raw_prev": [item["raw_prev"] for item in batch],
            "raw_next": [item["raw_next"] for item in batch],
            "id": [item["id"] for item in batch],
        }

        # Pad target sequences if available
        if "tgt" in batch[0]:
            tgt_list = [item["tgt"] for item in batch]
            tgt_padded = pad_sequence(
                tgt_list, batch_first=True, padding_value=Config.PAD_IDX
            )
            batch_data["tgt"] = tgt_padded
            batch_data["raw_after"] = [item["raw_after"] for item in batch]

        return batch_data
