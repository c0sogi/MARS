import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional, Dict, Any

from library.config import Config
from library.vocab import Vocabulary
from library.utils import get_logger

logger = get_logger("data")


class InterleavedDataset(Dataset):
    """
    Dataset that prepares interleaved sequences for the Global-Localization Transformer.
    Inserts [GAP] tokens between words and handles dynamic masking for training.
    """

    def __init__(
        self,
        split: str,
        vocab: Vocabulary,
        load_cached_data: bool = True,
        max_len: Optional[int] = None,
        debug: Optional[bool] = None,
        debug_size: Optional[int] = None,
    ):
        self.split = split
        self.vocab = vocab
        self.max_len = max_len if max_len is not None else Config.MAX_LEN
        self.debug = debug if debug is not None else Config.DEBUG
        self.debug_size = debug_size if debug_size is not None else Config.DEBUG_SIZE

        # Determine paths based on split
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.cache_path = Config.TRAIN_TOKENS_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.cache_path = Config.VAL_TOKENS_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
            self.cache_path = Config.TEST_TOKENS_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load data (cached or computed)
        self.data = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data: bool) -> pd.DataFrame:
        """
        Loads data from cache or computes it from metadata.
        Strictly follows the caching logic requirement.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            logger.info(f"Loading {self.split} tokens from cache: {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                logger.info(f"Loaded {len(df)} rows from cache.")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        logger.info(f"Processing {self.split} data from metadata...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        # Debug sampling
        if self.debug:
            logger.info(f"Debug mode: sampling {self.debug_size} rows.")
            df = df.head(self.debug_size)

        # Tokenize sentences
        logger.info("Tokenizing sentences...")
        sentences = df["sentence"].astype(str).tolist()
        token_ids_list = []

        for sent in sentences:
            # Simple whitespace tokenization
            tokens = sent.split()
            ids = self.vocab.encode_sequence(tokens)
            token_ids_list.append(ids)

        df["token_ids"] = token_ids_list

        # Normalize columns
        if "id" not in df.columns:
            df["id"] = df.index

        df = df[["id", "token_ids"]]

        # 3. Save to cache
        logger.info(f"Saving {self.split} tokens to cache: {self.cache_path}")
        df.to_parquet(self.cache_path, index=False)

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        token_ids = row["token_ids"]

        # Ensure token_ids is a list (pyarrow might return numpy array)
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        row_id = row["id"]

        if self.split == "test":
            return self._process_inference(token_ids, row_id)
        else:
            return self._process_training(token_ids, row_id)

    def _process_training(self, token_ids: List[int], row_id: int):
        # Filter short sentences: Need at least 3 words to remove a middle one
        if len(token_ids) < 3:
            return None

        # Select word to remove (index k).
        # Constraint: Never first (0) or last (len-1) word.
        # Range for k: [1, len(token_ids) - 2]
        k = np.random.randint(1, len(token_ids) - 1)

        target_word_id = token_ids[k]

        # Construct remaining sequence by removing word at k
        remaining_ids = token_ids[:k] + token_ids[k + 1 :]

        # Interleave with GAP tokens
        # Sequence: [w0, GAP, w1, GAP, ...]
        gap_id = self.vocab.gap_index
        interleaved_ids = []
        gap_mask = []  # 1 for GAP, 0 for word

        if remaining_ids:
            interleaved_ids.append(remaining_ids[0])
            gap_mask.append(0)

            for token in remaining_ids[1:]:
                interleaved_ids.append(gap_id)
                gap_mask.append(1)
                interleaved_ids.append(token)
                gap_mask.append(0)

        # Calculate target gap index
        # The removed word was between remaining_ids[k-1] and remaining_ids[k].
        # In the interleaved sequence:
        # Word at index i (0-based in remaining) is at position 2*i in interleaved.
        # The gap after word k-1 is at: 2*(k-1) + 1 = 2k - 1.
        target_loc_idx = 2 * k - 1

        # Handle Truncation
        if len(interleaved_ids) > self.max_len:
            interleaved_ids = interleaved_ids[: self.max_len]
            gap_mask = gap_mask[: self.max_len]

            # If target location is truncated, ignore this sample in loss
            if target_loc_idx >= self.max_len:
                target_loc_idx = -100
                target_word_id = -100

        return {
            "input_ids": torch.tensor(interleaved_ids, dtype=torch.long),
            "gap_mask": torch.tensor(gap_mask, dtype=torch.long),
            "target_loc": torch.tensor(target_loc_idx, dtype=torch.long),
            "target_id": torch.tensor(target_word_id, dtype=torch.long),
            "row_id": row_id,
        }

    def _process_inference(self, token_ids: List[int], row_id: int):
        # For test, words are already removed. Just interleave.
        gap_id = self.vocab.gap_index
        interleaved_ids = []
        gap_mask = []

        if token_ids:
            interleaved_ids.append(token_ids[0])
            gap_mask.append(0)
            for token in token_ids[1:]:
                interleaved_ids.append(gap_id)
                gap_mask.append(1)
                interleaved_ids.append(token)
                gap_mask.append(0)

        # Truncation
        if len(interleaved_ids) > self.max_len:
            interleaved_ids = interleaved_ids[: self.max_len]
            gap_mask = gap_mask[: self.max_len]

        return {
            "input_ids": torch.tensor(interleaved_ids, dtype=torch.long),
            "gap_mask": torch.tensor(gap_mask, dtype=torch.long),
            "target_loc": torch.tensor(-1, dtype=torch.long),  # Dummy
            "target_id": torch.tensor(-1, dtype=torch.long),  # Dummy
            "row_id": row_id,
        }


def collate_fn(batch):
    """
    Custom collate function to handle padding and filtering of None samples.
    """
    # Filter out None samples (short sentences)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    input_ids = [item["input_ids"] for item in batch]
    gap_masks = [item["gap_mask"] for item in batch]
    target_locs = [item["target_loc"] for item in batch]
    target_ids = [item["target_id"] for item in batch]
    row_ids = [item["row_id"] for item in batch]

    # Pad sequences
    # Note: Vocabulary ensures PAD token is at index 0.
    pad_idx = 0

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
    gap_masks_padded = pad_sequence(gap_masks, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids_padded,
        "gap_mask": gap_masks_padded,
        "target_loc": torch.stack(target_locs),
        "target_id": torch.stack(target_ids),
        "row_id": row_ids,
    }


def get_dataloaders(
    vocab: Vocabulary,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns DataLoaders for train, val, and test splits.
    """

    # Initialize Datasets
    train_ds = InterleavedDataset("train", vocab, load_cached_data)
    val_ds = InterleavedDataset("val", vocab, load_cached_data)
    test_ds = InterleavedDataset("test", vocab, load_cached_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
