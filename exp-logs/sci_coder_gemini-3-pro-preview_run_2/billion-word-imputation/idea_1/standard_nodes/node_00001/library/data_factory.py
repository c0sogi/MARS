import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    BatchEncoding,
)
from typing import Tuple, Dict, Optional
from library.config import Config

# Set seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class MLMDataset(Dataset):
    """
    A PyTorch Dataset for Masked Language Modeling.
    Wraps numpy arrays of input_ids and attention_masks to minimize memory overhead.
    """

    def __init__(self, input_ids: np.ndarray, attention_masks: np.ndarray):
        """
        Args:
            input_ids: Numpy array of shape (N, seq_len) containing token IDs.
            attention_masks: Numpy array of shape (N, seq_len) containing attention masks.
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Convert numpy types to torch tensors on the fly
        # input_ids for RoBERTa (vocab ~50k) fit in uint16, but torch needs Long (int64)
        return {
            "input_ids": torch.tensor(
                self.input_ids[idx].astype(np.int64), dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.attention_masks[idx].astype(np.int64), dtype=torch.long
            ),
        }


def prepare_tokenized_data(
    tokenizer: PreTrainedTokenizerBase, split: str, load_cached_data: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads raw data, tokenizes it, and caches the result as numpy arrays.

    Args:
        tokenizer: The HuggingFace tokenizer.
        split: 'train' or 'val'.
        load_cached_data: Whether to try loading from cache.

    Returns:
        Tuple of (input_ids, attention_masks) as numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    input_ids_path = os.path.join(Config.WORKING_DIR, f"{split}_input_ids.npy")
    attn_mask_path = os.path.join(Config.WORKING_DIR, f"{split}_attention_mask.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(input_ids_path) and os.path.exists(attn_mask_path):
            print(f"Loading cached {split} data from {Config.WORKING_DIR}...")
            try:
                input_ids = np.load(input_ids_path)
                attention_masks = np.load(attn_mask_path)
                return input_ids, attention_masks
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache not found for {split}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from source...")

    # Determine source path
    if split == "train":
        source_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        source_path = Config.VAL_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load raw text
    df = pd.read_parquet(source_path, columns=["sentence"])
    sentences = df["sentence"].tolist()

    # Tokenize
    # We use batch_encode_plus. For 24M rows, we might want to chunk,
    # but given 220GB RAM, we can likely do it in large batches or one go if careful.
    # To be safe and show progress (implicitly), we process in chunks.

    chunk_size = 100000
    all_input_ids = []
    all_attention_masks = []

    total_samples = len(sentences)

    # Pre-allocate numpy arrays to avoid memory fragmentation with lists
    # RoBERTa vocab < 65535, so uint16 is sufficient and saves RAM.
    shape = (total_samples, Config.MAX_SEQ_LEN)
    input_ids_np = np.empty(shape, dtype=np.uint16)
    attention_masks_np = np.empty(shape, dtype=np.uint8)  # Masks are 0 or 1

    print(f"Tokenizing {total_samples} sentences...")

    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        batch_sentences = sentences[i:end_idx]

        encodings = tokenizer(
            batch_sentences,
            add_special_tokens=True,
            max_length=Config.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,  # Return lists first
        )

        # Assign to pre-allocated arrays
        # Note: encodings['input_ids'] is a list of lists
        input_ids_np[i:end_idx] = encodings["input_ids"]
        attention_masks_np[i:end_idx] = encodings["attention_mask"]

    # 3. Save to cache
    print(f"Saving {split} tokenized data to cache...")
    np.save(input_ids_path, input_ids_np)
    np.save(attn_mask_path, attention_masks_np)

    return input_ids_np, attention_masks_np


def get_dataloaders(
    tokenizer: PreTrainedTokenizerBase, load_cached_data: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates Train and Validation DataLoaders.

    Args:
        tokenizer: The tokenizer to use for data processing and the collator.
        load_cached_data: Whether to use cached numpy files.

    Returns:
        (train_dataloader, val_dataloader)
    """

    # Prepare Data
    train_ids, train_masks = prepare_tokenized_data(
        tokenizer, "train", load_cached_data
    )
    val_ids, val_masks = prepare_tokenized_data(tokenizer, "val", load_cached_data)

    # Create Datasets
    train_dataset = MLMDataset(train_ids, train_masks)
    val_dataset = MLMDataset(val_ids, val_masks)

    # Define Collator
    # DataCollatorForLanguageModeling handles the dynamic masking (15% probability by default)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=data_collator,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=data_collator,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_dataloader, val_dataloader
