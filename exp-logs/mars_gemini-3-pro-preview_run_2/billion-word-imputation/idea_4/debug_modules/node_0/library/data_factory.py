import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Tuple, List, Dict, Any
from library.config import Config
from library.utils import set_seed

# -------------------------------------------------------------------------
# Data Loading & Caching Logic
# -------------------------------------------------------------------------


def load_data(
    metadata_path: str,
    cache_name: str,
    load_cached_data: bool = True,
    debug: bool = False,
    debug_size: int = 20000,
) -> pd.DataFrame:
    """
    Loads data from parquet metadata with caching and optional debug sampling.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    debug_suffix = f"_debug_{debug_size}" if debug else ""
    cache_file = os.path.join(Config.WORKING_DIR, f"{cache_name}{debug_suffix}.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"[{cache_name}] Loading cached data from {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"[{cache_name}] Cache load failed ({e}). Reloading from source.")

    # 2. Load from source metadata
    print(f"[{cache_name}] Loading raw data from {metadata_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)

    # 3. Apply Debug Sampling
    if debug:
        print(f"[{cache_name}] Sampling {debug_size} rows for debug mode.")
        if len(df) > debug_size:
            df = df.sample(n=debug_size, random_state=Config.SEED).reset_index(
                drop=True
            )

    # 4. Save to cache
    print(f"[{cache_name}] Saving processed data to {cache_file}")
    df.to_parquet(cache_file, index=False)

    return df


# -------------------------------------------------------------------------
# Dataset Implementations
# -------------------------------------------------------------------------


class LocatorDataset(Dataset):
    """
    Dataset for Stage 1 (Locator).
    Synthetically removes a word and labels the position of the gap.
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_len: int = 128
    ):
        self.sentences = df["sentence"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]

        # Tokenize with offset mapping to align tokens to characters
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        # Split text into words to select one for removal
        words = text.split()

        # Constraint: Never remove first or last word. Need at least 3 words.
        if len(words) < 3:
            return self._create_empty_sample(len(input_ids))

        # Select random word index (1 to len-2)
        remove_idx = np.random.randint(1, len(words) - 1)

        # Find character span of the selected word
        # We reconstruct spans manually to ensure alignment
        current_pos = 0
        word_spans = []
        for w in words:
            # Find next occurrence of word w
            start = text.find(w, current_pos)
            # If find fails (shouldn't happen with split), fallback
            if start == -1:
                start = current_pos
            end = start + len(w)
            word_spans.append((start, end))
            current_pos = end

        target_start_char, target_end_char = word_spans[remove_idx]

        # Identify tokens corresponding to this word
        tokens_to_remove = []
        for i, (off_start, off_end) in enumerate(offset_mapping):
            # Skip special tokens
            if off_start == 0 and off_end == 0:
                continue

            # Check for overlap. If token is within the word boundaries.
            # We use a loose overlap check: if token starts inside the word span
            if off_start >= target_start_char and off_end <= target_end_char:
                tokens_to_remove.append(i)

        if not tokens_to_remove:
            return self._create_empty_sample(len(input_ids))

        remove_start = min(tokens_to_remove)
        remove_end = max(tokens_to_remove)

        # Create new input_ids with tokens removed
        new_input_ids = input_ids[:remove_start] + input_ids[remove_end + 1 :]

        # The gap is after the token at index (remove_start - 1)
        label_idx = remove_start - 1

        return self._prepare_output(new_input_ids, label_idx)

    def _create_empty_sample(self, length):
        # Return a sample with no positive labels
        return self._prepare_output([self.tokenizer.pad_token_id] * length, -1)

    def _prepare_output(self, input_ids, label_idx):
        # Pad/Truncate to max_len
        curr_len = len(input_ids)
        pad_len = self.max_len - curr_len

        if pad_len < 0:
            input_ids = input_ids[: self.max_len]
            if label_idx >= self.max_len:
                label_idx = -1
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        # Create Binary Labels
        labels = np.zeros(self.max_len, dtype=np.float32)
        if 0 <= label_idx < self.max_len:
            labels[label_idx] = 1.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if i < curr_len else 0 for i in range(self.max_len)],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


class InFillerDataset(Dataset):
    """
    Dataset for Stage 2 (In-Filler).
    Masks a random word and provides the token ID as label.
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_len: int = 128
    ):
        self.sentences = df["sentence"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        words = text.split()
        if len(words) < 3:
            return self._prepare_output(input_ids, -1, -1)

        # Select random word
        target_idx = np.random.randint(1, len(words) - 1)

        # Find spans
        current_pos = 0
        word_spans = []
        for w in words:
            start = text.find(w, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(w)
            word_spans.append((start, end))
            current_pos = end

        target_start, target_end = word_spans[target_idx]

        # Identify tokens
        target_tokens = []
        for i, (off_start, off_end) in enumerate(offset_mapping):
            if off_start == 0 and off_end == 0:
                continue
            if off_start >= target_start and off_end <= target_end:
                target_tokens.append(i)

        if not target_tokens:
            return self._prepare_output(input_ids, -1, -1)

        mask_start = min(target_tokens)
        mask_end = max(target_tokens)

        # Replace whole word span with ONE mask token
        # Label is the first token of the word (best approx for single-mask prediction)
        target_label_id = input_ids[mask_start]

        new_input_ids = (
            input_ids[:mask_start]
            + [self.tokenizer.mask_token_id]
            + input_ids[mask_end + 1 :]
        )

        return self._prepare_output(new_input_ids, mask_start, target_label_id)

    def _prepare_output(self, input_ids, mask_idx, label_id):
        curr_len = len(input_ids)
        pad_len = self.max_len - curr_len

        if pad_len < 0:
            input_ids = input_ids[: self.max_len]
            if mask_idx >= self.max_len:
                mask_idx = -1
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        labels = np.full(self.max_len, -100, dtype=np.int64)
        if 0 <= mask_idx < self.max_len:
            labels[mask_idx] = label_id

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if i < curr_len else 0 for i in range(self.max_len)],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TestDataset(Dataset):
    """
    Dataset for Inference. Returns IDs and Raw Text.
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_len: int = 128
    ):
        self.ids = df["id"].tolist()
        self.sentences = df["sentence"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        row_id = self.ids[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "id": row_id,
            "original_text": text,
        }


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------


def create_dataloaders(
    load_cached_data: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for Locator (Train/Val), InFiller (Train/Val), and Test.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    df_train = load_data(
        Config.TRAIN_METADATA,
        "train_cache",
        load_cached_data,
        Config.DEBUG,
        Config.DEBUG_SIZE,
    )
    df_val = load_data(
        Config.VAL_METADATA,
        "val_cache",
        load_cached_data,
        Config.DEBUG,
        Config.DEBUG_SIZE,
    )
    df_test = load_data(Config.TEST_METADATA, "test_cache", load_cached_data, False)

    # 2. Initialize Tokenizers
    locator_tokenizer = AutoTokenizer.from_pretrained(Config.LOCATOR_MODEL)
    infiller_tokenizer = AutoTokenizer.from_pretrained(Config.INFILLER_MODEL)

    # 3. Instantiate Datasets
    print("Building Locator Datasets...")
    train_loc_ds = LocatorDataset(df_train, locator_tokenizer, Config.MAX_LEN)
    val_loc_ds = LocatorDataset(df_val, locator_tokenizer, Config.MAX_LEN)

    print("Building In-Filler Datasets...")
    train_fill_ds = InFillerDataset(df_train, infiller_tokenizer, Config.MAX_LEN)
    val_fill_ds = InFillerDataset(df_val, infiller_tokenizer, Config.MAX_LEN)

    print("Building Test Dataset...")
    test_ds = TestDataset(df_test, locator_tokenizer, Config.MAX_LEN)

    # 4. Create DataLoaders
    train_loc_loader = DataLoader(
        train_loc_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loc_loader = DataLoader(
        val_loc_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    train_fill_loader = DataLoader(
        train_fill_ds,
        batch_size=Config.INFILLER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_fill_loader = DataLoader(
        val_fill_ds,
        batch_size=Config.INFILLER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return (
        train_loc_loader,
        val_loc_loader,
        train_fill_loader,
        val_fill_loader,
        test_loader,
    )
