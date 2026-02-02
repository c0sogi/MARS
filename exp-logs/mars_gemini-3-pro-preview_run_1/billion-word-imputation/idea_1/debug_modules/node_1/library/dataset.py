import os
import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import logger
from library.vocab import Vocabulary


class InfillingDataset(Dataset):
    """
    Dataset class for the Sentence Infilling Task.
    Handles loading, tokenization, and dynamic gap generation for training.
    """

    def __init__(self, split, vocab, load_cached_data=True, debug_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            vocab (Vocabulary): Vocabulary instance.
            load_cached_data (bool): Whether to use cached tokenized data.
            debug_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.vocab = vocab
        self.load_cached_data = load_cached_data
        self.debug_size = debug_size

        # Determine paths based on split
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Cache file path
        self.cache_path = os.path.join(Config.WORKING_DIR, f"{split}_tokens.parquet")

        # For test set, we need to ensure IDs are available
        self.ids = None

        # Load data (tokens and optional IDs)
        self.data = self._load_data()

        if split == "test":
            # IDs are loaded into self.ids during _load_data
            if self.ids is None:
                raise RuntimeError("IDs not loaded for test set.")

    def _load_data(self):
        """
        Loads data from cache or computes it from metadata.
        Returns a list of tokenized sequences (List[List[int]]).
        For test set, populates self.ids.
        """
        # 1. Try Loading from Cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            logger.info(f"[{self.split}] Loading cached data from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)

                if self.debug_size:
                    df = df.iloc[: self.debug_size]

                if self.split == "test":
                    self.ids = df["id"].values

                # Return tokens column as list
                return df["tokens"].tolist()
            except Exception as e:
                logger.warning(
                    f"[{self.split}] Failed to load cache: {e}. Recomputing..."
                )

        # 2. Compute from Scratch
        logger.info(f"[{self.split}] Processing data from {self.metadata_path}")

        try:
            df = pd.read_csv(self.metadata_path)
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            raise

        if self.debug_size:
            df = df.iloc[: self.debug_size]

        # Tokenize
        logger.info(f"[{self.split}] Tokenizing {len(df)} sentences...")

        # Optimization: Pre-fetch vocab attributes
        stoi = self.vocab.stoi
        unk_id = self.vocab.unk_token_id

        def tokenize_func(sentence):
            if not isinstance(sentence, str):
                return []
            # Simple whitespace splitting
            words = sentence.split()
            return [stoi.get(w, unk_id) for w in words]

        # Apply tokenization
        sentences = df["sentence"].fillna("").astype(str).tolist()
        tokenized_data = [tokenize_func(s) for s in sentences]

        # Create DataFrame for caching
        cache_df = pd.DataFrame({"tokens": tokenized_data})

        if self.split == "test":
            self.ids = df["id"].values
            cache_df["id"] = self.ids

        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        logger.info(f"[{self.split}] Saving cache to {self.cache_path}")
        cache_df.to_parquet(self.cache_path, index=False)

        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'input_ids': torch.LongTensor,
                'targets': torch.LongTensor (only for train/val),
                'id': int (only for test)
            }
        """
        tokens = self.data[idx]

        # Ensure tokens is a list (parquet might return numpy array)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        if self.split == "test":
            # Test mode: No removal, just wrap with START/END
            input_ids = (
                [self.vocab.stoi[self.vocab.TOKEN_START]]
                + tokens
                + [self.vocab.stoi[self.vocab.TOKEN_END]]
            )
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "id": self.ids[idx],
            }

        else:
            # Train/Val mode: Randomly remove one word
            # Constraint: Never first or last word.
            n_tokens = len(tokens)

            # We need at least 3 tokens to remove a middle one safely
            if n_tokens < 3:
                remove_idx = -1
            else:
                # Random index between 1 and n_tokens - 2 (inclusive)
                # np.random.randint(low, high) excludes high, so use n_tokens - 1
                remove_idx = np.random.randint(1, n_tokens - 1)

            if remove_idx != -1:
                target_word_id = tokens[remove_idx]

                # Construct input: remove the word at remove_idx
                input_tokens = tokens[:remove_idx] + tokens[remove_idx + 1 :]

                # Construct targets
                # Length of input_ids will be len(input_tokens) + 2 (START/END)
                # We predict the gap AFTER each token.
                target_len = len(input_tokens) + 2
                targets = [self.vocab.no_insert_token_id] * target_len

                # Logic:
                # We prepend [START] at index 0.
                # The token at input_ids[remove_idx] corresponds to the token *before* the removed gap.
                # Therefore, the target for input_ids[remove_idx] is the removed word.
                targets[remove_idx] = target_word_id

                input_ids = (
                    [self.vocab.stoi[self.vocab.TOKEN_START]]
                    + input_tokens
                    + [self.vocab.stoi[self.vocab.TOKEN_END]]
                )

            else:
                # Fallback for very short sentences: no removal
                input_ids = (
                    [self.vocab.stoi[self.vocab.TOKEN_START]]
                    + tokens
                    + [self.vocab.stoi[self.vocab.TOKEN_END]]
                )
                targets = [self.vocab.no_insert_token_id] * len(input_ids)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "targets": torch.tensor(targets, dtype=torch.long),
            }

    def collate_fn(self, batch):
        """
        Custom collate function to pad sequences.
        """
        input_ids = [item["input_ids"] for item in batch]

        # Pad inputs with PAD token
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.vocab.pad_token_id
        )

        result = {"input_ids": input_ids_padded}

        if "targets" in batch[0]:
            targets = [item["targets"] for item in batch]
            # Pad targets with -100 (ignore index for CrossEntropyLoss)
            targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
            result["targets"] = targets_padded

        if "id" in batch[0]:
            ids = [item["id"] for item in batch]
            result["id"] = torch.tensor(ids, dtype=torch.long)

        return result


def get_dataloaders(vocab, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test splits.

    Args:
        vocab (Vocabulary): Vocabulary instance.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    debug_size = Config.DEBUG_SAMPLE_SIZE

    # Train Set
    train_dataset = InfillingDataset(
        split="train",
        vocab=vocab,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
    )

    # Val Set
    val_dataset = InfillingDataset(
        split="val",
        vocab=vocab,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
    )

    # Test Set
    test_dataset = InfillingDataset(
        split="test",
        vocab=vocab,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
    )

    return train_loader, val_loader, test_loader
