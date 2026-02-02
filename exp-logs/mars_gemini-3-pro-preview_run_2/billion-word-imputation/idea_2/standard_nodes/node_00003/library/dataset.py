import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.tokenizer import get_tokenizer


class MissingWordDataset(Dataset):
    """
    Dataset class for the Missing Word Insertion task.

    Modes:
    - 'train'/'val': Synthetically removes one word from complete sentences.
      Returns: input_ids, loc_target, word_target, gap_idx
    - 'test': Processes sentences with an existing missing word.
      Returns: id, input_ids
    """

    def __init__(self, data, tokenizer, max_len, mode="train"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Pre-fetch special token IDs
        self.unk_token_id = tokenizer.unk_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row["sentence"]

        if self.mode == "test":
            # In test mode, the word is already removed.
            # We just need to encode the sequence.
            input_ids = self.tokenizer.encode(sentence, max_len=self.max_len)

            return {
                "id": row["id"],
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
            }

        else:
            # Train/Val mode: Synthetic Data Generation
            words = sentence.split()
            n_words = len(words)

            # Safety check: we need at least 3 words to remove one from the middle
            # (indices 1 to n-2). The data loader should filter this, but we handle edge cases.
            if n_words < 3:
                # Fallback: return the item at index 0 (assuming it's valid)
                # or a dummy sample if idx 0 is also invalid (unlikely after filtering).
                if idx != 0:
                    return self.__getitem__(0)
                else:
                    # Extreme fallback for index 0 if invalid
                    return self._get_dummy_sample()

            # Randomly select a word to remove.
            # Constraints: Never the first (idx 0) or last (idx n-1) word.
            # Eligible indices: 1 to n-2.
            remove_idx = random.randint(1, n_words - 2)
            target_word = words[remove_idx]

            # Create the input sequence by removing the word
            input_words = words[:remove_idx] + words[remove_idx + 1 :]
            input_sentence = " ".join(input_words)

            # Tokenize input
            input_ids = self.tokenizer.encode(input_sentence, max_len=self.max_len)

            # Calculate Gap Index
            # The gap is located after the word at `remove_idx - 1` in the original list.
            # In the new `input_words` list, this corresponds exactly to index `remove_idx - 1`.
            # Example: [A, B, C]. Remove B (idx 1). Input [A, C]. Gap after A (idx 0).
            # remove_idx = 1. gap_idx = 0.
            gap_idx = remove_idx - 1

            # Create Location Target (Binary Vector)
            loc_target = torch.zeros(self.max_len, dtype=torch.float)

            # Handle truncation edge cases
            if gap_idx < self.max_len:
                loc_target[gap_idx] = 1.0
                target_word_id = self.tokenizer.word2idx.get(
                    target_word, self.unk_token_id
                )
                safe_gap_idx = gap_idx
            else:
                # If gap is outside truncated sequence:
                # 1. Use a safe index (0) to prevent CUDA indexing errors in Trainer.
                # 2. Set target to -100 so CrossEntropyLoss ignores this sample.
                target_word_id = -100
                safe_gap_idx = 0

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "loc_target": loc_target,
                "word_target": torch.tensor(target_word_id, dtype=torch.long),
                "gap_idx": torch.tensor(safe_gap_idx, dtype=torch.long),
            }

    def _get_dummy_sample(self):
        """Returns a zero-tensor dummy sample to prevent crashing on empty/invalid data."""
        return {
            "input_ids": torch.zeros(self.max_len, dtype=torch.long),
            "loc_target": torch.zeros(self.max_len, dtype=torch.float),
            "word_target": torch.tensor(self.unk_token_id, dtype=torch.long),
            "gap_idx": torch.tensor(0, dtype=torch.long),
        }


def _load_and_process_data(path, mode, config, load_cached_data):
    """
    Loads data from parquet, applies filtering for train/val sets,
    and handles caching of the processed dataframe.
    """
    # Construct a cache filename that includes debug state to avoid collisions
    debug_suffix = (
        f"_debug_{config.DEBUG_SAMPLE_SIZE}" if config.DEBUG_SAMPLE_SIZE else ""
    )
    cache_filename = f"{mode}_filtered{debug_suffix}.parquet"
    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing...")

    # 2. Process from Scratch
    print(f"Loading raw {mode} data from {path}...")
    df = pd.read_parquet(path)

    # Apply Debug Sampling if configured
    if config.DEBUG_SAMPLE_SIZE is not None:
        if len(df) > config.DEBUG_SAMPLE_SIZE:
            print(f"Sampling {config.DEBUG_SAMPLE_SIZE} rows for {mode}...")
            df = df.iloc[: config.DEBUG_SAMPLE_SIZE]

    # Apply Filtering for Train/Val (Sentence Length >= 3)
    if mode in ["train", "val"]:
        print(f"Filtering {mode} data for minimum word count...")
        # We need at least 3 words to be able to remove one from the middle.
        # Using vectorized string operations for efficiency.
        # Note: This assumes space-separated tokens, consistent with the tokenizer logic.
        mask = df["sentence"].str.split().str.len() >= 3
        original_len = len(df)
        df = df[mask].reset_index(drop=True)
        print(f"Filtered {mode}: {original_len} -> {len(df)} rows.")

    # 3. Save to Cache
    print(f"Saving processed {mode} data to {cache_path}")
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(config=Config, load_cached_data=True):
    """
    Main function to initialize datasets and dataloaders.

    Args:
        config: Configuration class.
        load_cached_data (bool): Whether to use cached filtered dataframes.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # Set seeds for reproducibility in data processing
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # 1. Initialize Tokenizer
    tokenizer = get_tokenizer(load_cached_data=load_cached_data)

    # 2. Load and Process Data
    df_train = _load_and_process_data(
        config.TRAIN_DATA_PATH, "train", config, load_cached_data
    )

    df_val = _load_and_process_data(
        config.VAL_DATA_PATH, "val", config, load_cached_data
    )

    # Test data doesn't require filtering logic, but we use the helper for consistency (e.g. debug sampling)
    df_test = _load_and_process_data(
        config.TEST_DATA_PATH, "test", config, load_cached_data
    )

    # 3. Create Datasets
    train_dataset = MissingWordDataset(
        df_train, tokenizer, config.MAX_SEQ_LEN, mode="train"
    )
    val_dataset = MissingWordDataset(df_val, tokenizer, config.MAX_SEQ_LEN, mode="val")
    test_dataset = MissingWordDataset(
        df_test, tokenizer, config.MAX_SEQ_LEN, mode="test"
    )

    # 4. Create DataLoaders
    # Pin memory enables faster data transfer to CUDA devices
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, tokenizer
