import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.vocab import Vocabulary
from library.utils import setup_logger

logger = setup_logger("data")


def tokenize_and_cache(
    metadata_path: str,
    vocab: Vocabulary,
    cache_path: str,
    load_cached_data: bool = True,
) -> pd.DataFrame:
    """
    Reads the metadata CSV, tokenizes sentences into integer lists,
    and caches the result as a Parquet file.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached tokens from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Read CSV in chunks to handle large files efficiently
    chunk_size = 500_000
    data_list = []

    try:
        reader = pd.read_csv(metadata_path, chunksize=chunk_size)
        total_processed = 0

        for chunk in reader:
            if "sentence" not in chunk.columns:
                continue

            # Drop NaNs
            chunk = chunk.dropna(subset=["sentence"])

            # Extract lists
            ids = chunk["id"].tolist()
            sentences = chunk["sentence"].astype(str).tolist()

            # Tokenize and Numericalize
            batch_token_ids = []
            for s in sentences:
                tokens = s.split()
                token_ids = vocab.numericalize(tokens)
                batch_token_ids.append(token_ids)

            # Store
            for i, t_ids in zip(ids, batch_token_ids):
                data_list.append({"id": i, "token_ids": t_ids})

            total_processed += len(chunk)
            if total_processed % 2_000_000 == 0:
                logger.info(f"Tokenized {total_processed} sentences...")

    except Exception as e:
        logger.error(f"Error processing metadata: {e}")
        raise e

    df = pd.DataFrame(data_list)

    # 3. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        logger.info(f"Saving tokenized data to {cache_path}...")
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    return df


class InfillingDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        vocab: Vocabulary,
        mode: str = "train",
        cache_path: str = None,
        load_cached_data: bool = True,
        max_len: int = Config.MAX_SEQ_LEN,
    ):
        """
        Args:
            metadata_path: Path to the raw CSV.
            vocab: Loaded Vocabulary object.
            mode: 'train', 'val', or 'test'.
            cache_path: Path to save/load parquet cache.
            load_cached_data: Whether to use cache.
            max_len: Max sequence length.
        """
        self.vocab = vocab
        self.mode = mode
        self.max_len = max_len

        # Determine cache path if not provided
        if cache_path is None:
            filename = f"{mode}_tokens.parquet"
            cache_path = os.path.join(Config.WORK_DIR, filename)

        # Load data
        self.df = tokenize_and_cache(metadata_path, vocab, cache_path, load_cached_data)

        # For training/validation, we need sentences with at least 3 words
        # (Start, Gap_Candidate, End) to perform valid masking.
        if self.mode in ["train", "val"]:
            initial_len = len(self.df)
            self.df = self.df[self.df["token_ids"].apply(len) >= 3].reset_index(
                drop=True
            )
            if len(self.df) < initial_len:
                logger.info(f"Filtered {initial_len - len(self.df)} short sentences.")

        # Cache special token indices for speed
        self.no_insert_idx = vocab.get_no_insert_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        token_ids = list(row["token_ids"])
        row_id = row["id"]

        # Truncate to max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len]

        L = len(token_ids)

        if self.mode in ["train", "val"]:
            # Dynamic Masking Logic
            # We select a word to remove. Constraints: Not first (0), not last (L-1).
            # Available indices for removal: 1 to L-2.

            # Safety check
            if L < 3:
                # Should be filtered, but fallback
                input_ids = token_ids
                targets = [self.no_insert_idx] * len(input_ids)
            else:
                # Pick random index to remove
                remove_idx = np.random.randint(1, L - 1)
                removed_token = token_ids[remove_idx]

                # Create Input: Remove the token
                input_ids = token_ids[:remove_idx] + token_ids[remove_idx + 1 :]

                # Create Targets
                # Initialize all as NO_INSERT
                targets = [self.no_insert_idx] * len(input_ids)

                # The gap is located after the token at (remove_idx - 1) in the new sequence.
                # Example: [A, B, C, D]. Remove B (idx 1).
                # Input: [A, C, D].
                # Gap is between A and C.
                # A is at index 0. remove_idx - 1 = 0.
                # Target at 0 is B.
                target_pos = remove_idx - 1
                targets[target_pos] = removed_token

        else:
            # Test Mode
            # Sentence already has word removed.
            input_ids = token_ids
            targets = [self.no_insert_idx] * len(input_ids)

        return {
            "id": row_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Pads input_ids and targets to the maximum length in the batch.
    """
    ids = [item["id"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    targets = [item["targets"] for item in batch]

    # Hardcoded indices based on library/vocab.py structure:
    # special_tokens = [PAD, UNK, NO_INSERT]
    # PAD = 0, NO_INSERT = 2
    pad_idx = 0
    no_insert_idx = 2

    # Pad inputs with PAD token
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)

    # Pad targets with NO_INSERT token
    # This ensures the model learns not to insert words after padding.
    targets_padded = pad_sequence(
        targets, batch_first=True, padding_value=no_insert_idx
    )

    # Create mask (optional, but length is useful)
    lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)

    return {
        "id": torch.tensor(ids, dtype=torch.long),
        "input_ids": input_ids_padded,
        "targets": targets_padded,
        "lengths": lengths,
    }


def get_dataloaders(
    vocab: Vocabulary,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
    debug: bool = False,
):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    logger.info("Initializing DataLoaders...")

    # Train
    train_ds = InfillingDataset(
        metadata_path=Config.TRAIN_METADATA,
        vocab=vocab,
        mode="train",
        load_cached_data=load_cached_data,
    )
    if debug:
        train_ds.df = train_ds.df.iloc[:5000]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Val
    val_ds = InfillingDataset(
        metadata_path=Config.VAL_METADATA,
        vocab=vocab,
        mode="val",
        load_cached_data=load_cached_data,
    )
    if debug:
        val_ds.df = val_ds.df.iloc[:1000]

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test
    test_ds = InfillingDataset(
        metadata_path=Config.TEST_METADATA,
        vocab=vocab,
        mode="test",
        load_cached_data=load_cached_data,
    )
    if debug:
        test_ds.df = test_ds.df.iloc[:1000]

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
