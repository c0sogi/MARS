import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("dataset")


def prepare_tokenized_data(split, vocab, load_cached_data=True):
    """
    Loads raw data, tokenizes it using the provided vocabulary, and caches it to Parquet.

    Args:
        split (str): One of 'train', 'val', 'test'.
        vocab (Vocabulary): The vocabulary instance for encoding.
        load_cached_data (bool): If True, attempts to load from existing parquet file.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'token_ids'.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_tokens.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {split} data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Rebuilding from scratch...")

    # 2. Process from scratch
    logger.info(f"Processing {split} data...")

    # Identify source file
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    # Debug Sampling
    if Config.DEBUG and split != "test":
        logger.info(
            f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows from {split}."
        )
        if len(df) > Config.DEBUG_SAMPLE_SIZE:
            df = df.sample(n=Config.DEBUG_SAMPLE_SIZE, random_state=42).reset_index(
                drop=True
            )

    # Tokenization
    logger.info(f"Tokenizing {len(df)} sentences...")

    # Optimization: Pre-fetch dictionary and method to avoid lookup overhead in loop
    stoi = vocab.stoi
    unk_idx = Config.UNK_IDX

    # Ensure sentences are strings and handle NaNs
    sentences = df["sentence"].fillna("").astype(str).tolist()

    tokenized_ids = []
    for s in sentences:
        tokens = s.split()
        # Map tokens to IDs, using UNK_IDX for missing words
        ids = [stoi.get(t, unk_idx) for t in tokens]
        tokenized_ids.append(ids)

    df["token_ids"] = tokenized_ids

    # Select only necessary columns to save space
    out_df = df[["id", "token_ids"]]

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    out_df.to_parquet(cache_path, index=False)
    logger.info(f"Saved tokenized data to {cache_path}")

    return out_df


class GapTokenDataset(Dataset):
    def __init__(self, data_df, mode="train"):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing 'token_ids'.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data_df
        self.mode = mode
        self.gap_token_id = Config.GAP_IDX

        # Calculate max words allowed to ensure interleaved sequence fits in MAX_SEQ_LEN
        # Interleaved length = 2 * n_words + 1
        # Therefore: n_words <= (MAX_SEQ_LEN - 1) / 2
        self.max_word_len = (Config.MAX_SEQ_LEN - 1) // 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row["token_ids"]  # List[int]

        # Truncate sequence if it exceeds maximum capacity
        if len(tokens) > self.max_word_len:
            tokens = tokens[: self.max_word_len]

        input_word_ids = []
        target_word_id = -100  # Default ignore index for CrossEntropy
        target_gap_idx = -1  # Default ignore index

        if self.mode in ["train", "val"]:
            n = len(tokens)
            # We need at least 3 tokens to remove a word that is NOT first or last.
            # Indices: 0, 1, ..., n-1. Valid removal: 1, ..., n-2.
            if n >= 3:
                # Randomly select a word to remove (excluding first and last)
                remove_idx = torch.randint(low=1, high=n - 1, size=(1,)).item()

                target_word_id = tokens[remove_idx]

                # Construct input by removing the selected word
                input_word_ids = tokens[:remove_idx] + tokens[remove_idx + 1 :]

                # The gap index corresponds to the removal index.
                # Example: [A, B, C]. Remove B (idx 1). Result: [A, C].
                # Interleaved: [G, A, G, C, G].
                # B was between A and C. That is the gap at index 1 in the gap sequence.
                target_gap_idx = remove_idx
            else:
                # Sequence too short to satisfy constraints.
                # Pass through without removal (no loss will be computed for this sample ideally)
                input_word_ids = tokens
                target_word_id = -100
                target_gap_idx = -1
        else:
            # Test mode: The sentence already has a word removed.
            input_word_ids = tokens
            target_word_id = 0  # Dummy
            target_gap_idx = 0  # Dummy

        # Construct Interleaved Sequence: [GAP] w1 [GAP] w2 ... [GAP]
        interleaved_ids = []
        token_type_ids = []  # 0 for Word, 1 for Gap

        # Start with initial GAP
        interleaved_ids.append(self.gap_token_id)
        token_type_ids.append(1)

        for w_id in input_word_ids:
            interleaved_ids.append(w_id)
            token_type_ids.append(0)

            interleaved_ids.append(self.gap_token_id)
            token_type_ids.append(1)

        return {
            "input_ids": torch.tensor(interleaved_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_word_id": torch.tensor(target_word_id, dtype=torch.long),
            "target_gap_idx": torch.tensor(target_gap_idx, dtype=torch.long),
            "original_id": row["id"],
        }


def collate_fn(batch):
    """
    Pads the batch of sequences to the maximum length in the batch.
    """
    input_ids_list = [item["input_ids"] for item in batch]
    token_type_ids_list = [item["token_type_ids"] for item in batch]
    target_word_ids = torch.stack([item["target_word_id"] for item in batch])
    target_gap_idxs = torch.stack([item["target_gap_idx"] for item in batch])
    original_ids = [item["original_id"] for item in batch]

    # Pad sequences
    # padding_value for input_ids is PAD_IDX
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=Config.PAD_IDX
    )

    # padding_value for token_type_ids is arbitrary for pads, but 0 is safe
    padded_token_type_ids = torch.nn.utils.rnn.pad_sequence(
        token_type_ids_list, batch_first=True, padding_value=0
    )

    # Create Attention Mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_input_ids != Config.PAD_IDX).long()

    return {
        "input_ids": padded_input_ids,
        "token_type_ids": padded_token_type_ids,
        "attention_mask": attention_mask,
        "target_word_id": target_word_ids,
        "target_gap_idx": target_gap_idxs,
        "original_ids": original_ids,
    }


def get_dataloaders(vocab, load_cached_data=True):
    """
    Factory function to create DataLoaders for all splits.

    Args:
        vocab (Vocabulary): Vocabulary object.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Prepare Data
    train_df = prepare_tokenized_data("train", vocab, load_cached_data)
    val_df = prepare_tokenized_data("val", vocab, load_cached_data)
    test_df = prepare_tokenized_data("test", vocab, load_cached_data)

    # Create Datasets
    train_ds = GapTokenDataset(train_df, mode="train")
    val_ds = GapTokenDataset(val_df, mode="val")
    test_ds = GapTokenDataset(test_df, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
