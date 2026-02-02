import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.vocab import load_or_build_artifacts


class GapTokenDataset(Dataset):
    def __init__(self, split="train", vocab=None, pos_map=None, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            vocab (Vocabulary): Vocabulary instance.
            pos_map (np.array): Array mapping word_id -> pos_id.
            debug (bool): If True, limits dataset size.
        """
        self.split = split
        self.vocab = vocab
        self.pos_map = pos_map
        self.debug = debug

        # Load tokenized data
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads tokenized data from cache or creates it from metadata.
        Returns a list of token_id lists (train/val) or list of dicts (test).
        """
        # Determine paths based on split
        if self.split == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
            cache_path = Config.TRAIN_TOKENS_PATH
        elif self.split == "val":
            metadata_path = Config.VAL_METADATA_PATH
            cache_path = Config.VAL_TOKENS_PATH
        elif self.split == "test":
            metadata_path = Config.TEST_METADATA_PATH
            cache_path = Config.TEST_TOKENS_PATH
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Check cache
        if os.path.exists(cache_path):
            try:
                # Load from parquet
                df = pd.read_parquet(cache_path)

                # Apply debug limit
                if self.debug or Config.DEBUG:
                    df = df.head(Config.DEBUG_SIZE)

                # Return appropriate format
                if self.split == "test":
                    return df[["id", "token_ids"]].to_dict("records")
                else:
                    return df["token_ids"].tolist()
            except Exception as e:
                print(f"Error loading cache {cache_path}: {e}. Rebuilding.")

        # Build from scratch
        print(f"Processing {self.split} data from {metadata_path}...")
        df = pd.read_csv(metadata_path)

        # Apply debug limit before processing to save time
        if (self.debug or Config.DEBUG) and len(df) > Config.DEBUG_SIZE:
            df = df.iloc[: Config.DEBUG_SIZE]

        # Tokenize function
        def tokenize(text):
            if not isinstance(text, str):
                return []
            return [self.vocab[t] for t in text.split()]

        # Apply tokenization
        df["token_ids"] = df["sentence"].apply(tokenize)

        # Filter invalid sentences for training/validation
        if self.split != "test":
            # Must have at least 3 tokens: [Word, Target, Period] to allow valid removal
            # Prompt says: "never the first or last word".
            df = df[df["token_ids"].apply(len) >= 3]

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if self.split == "test":
            save_df = df[["id", "token_ids"]]
            save_df.to_parquet(cache_path)
            return save_df.to_dict("records")
        else:
            save_df = df[["token_ids"]]
            save_df.to_parquet(cache_path)
            return save_df["token_ids"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "test":
            record = self.data[idx]
            token_ids = record["token_ids"]
            row_id = record["id"]

            # Construct interleaved sequence for test
            # [SOS] [GAP] w1 [GAP] w2 ... [GAP] wn [GAP] [EOS]
            input_ids = [Config.SOS_IDX, Config.GAP_IDX]

            for t in token_ids:
                input_ids.append(t)
                input_ids.append(Config.GAP_IDX)

            input_ids.append(Config.EOS_IDX)

            # Truncate
            if len(input_ids) > Config.MAX_SEQ_LEN:
                input_ids = input_ids[: Config.MAX_SEQ_LEN]

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "row_id": row_id,
            }

        else:
            # Train / Val
            token_ids = self.data[idx]
            seq_len = len(token_ids)

            # Select word to remove
            # Constraint: Not first (0), Not last (seq_len-1).
            # Indices eligible: 1 to seq_len - 2.

            if seq_len < 3:
                # Should have been filtered, but safety fallback
                remove_idx = 1 if seq_len > 1 else 0
            else:
                if self.split == "val":
                    # Deterministic selection for validation
                    low = 1
                    high = seq_len - 2
                    if high < low:
                        high = low
                    # Seed with index to be consistent across epochs
                    rng = np.random.RandomState(idx)
                    remove_idx = rng.randint(low, high + 1)
                else:
                    # Random selection for training
                    low = 1
                    high = seq_len - 2
                    if high < low:
                        high = low
                    remove_idx = random.randint(low, high + 1)

            target_word_id = token_ids[remove_idx]
            # Map word ID to POS ID
            target_pos_id = (
                self.pos_map[target_word_id] if self.pos_map is not None else 0
            )

            # Construct sequence: Remove the word and interleave GAPs
            remaining_tokens = token_ids[:remove_idx] + token_ids[remove_idx + 1 :]

            input_ids = [Config.SOS_IDX, Config.GAP_IDX]

            # Calculate target gap index
            # The gap corresponding to the removed word at `remove_idx`
            # is located at `2 * remove_idx + 1` in the interleaved sequence.
            target_gap_idx = 2 * remove_idx + 1

            for t in remaining_tokens:
                input_ids.append(t)
                input_ids.append(Config.GAP_IDX)

            input_ids.append(Config.EOS_IDX)

            # Truncate
            if len(input_ids) > Config.MAX_SEQ_LEN:
                input_ids = input_ids[: Config.MAX_SEQ_LEN]

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "target_gap_idx": target_gap_idx,
                "target_word_id": target_word_id,
                "target_pos_id": target_pos_id,
            }


def collate_fn(batch):
    """
    Pads sequences and constructs target tensors.
    """
    # Check if test batch (has row_id)
    if "row_id" in batch[0]:
        input_ids = [item["input_ids"] for item in batch]
        row_ids = [item["row_id"] for item in batch]

        # Pad inputs
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=Config.PAD_IDX
        )

        # Masks
        attention_mask = (input_ids_padded != Config.PAD_IDX).long()
        gap_mask = (input_ids_padded == Config.GAP_IDX).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "gap_mask": gap_mask,
            "row_ids": row_ids,
        }
    else:
        # Train/Val batch
        input_ids = [item["input_ids"] for item in batch]
        target_gap_idxs = [item["target_gap_idx"] for item in batch]
        target_word_ids = [item["target_word_id"] for item in batch]
        target_pos_ids = [item["target_pos_id"] for item in batch]

        # Pad inputs
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=Config.PAD_IDX
        )
        batch_size, seq_len = input_ids_padded.shape

        # Masks
        attention_mask = (input_ids_padded != Config.PAD_IDX).long()
        gap_mask = (input_ids_padded == Config.GAP_IDX).long()

        # Create dense targets
        # loc_targets: 1.0 at correct gap, 0.0 elsewhere (masked by gap_mask in loss usually)
        loc_targets = torch.zeros((batch_size, seq_len), dtype=torch.float)

        # syntax/word targets: ID at correct gap, -100 elsewhere
        syntax_targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        word_targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        for i in range(batch_size):
            gap_idx = target_gap_idxs[i]
            # Ensure target is within truncated sequence
            if gap_idx < seq_len:
                loc_targets[i, gap_idx] = 1.0
                syntax_targets[i, gap_idx] = target_pos_ids[i]
                word_targets[i, gap_idx] = target_word_ids[i]

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "gap_mask": gap_mask,
            "loc_targets": loc_targets,
            "syntax_targets": syntax_targets,
            "word_targets": word_targets,
        }


def get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False):
    """
    Creates DataLoaders for training and validation.
    """
    # Load artifacts (Vocab, POS Map)
    vocab, pos_map, _ = load_or_build_artifacts(load_cached_data=True)

    # Create Datasets
    train_ds = GapTokenDataset("train", vocab, pos_map, debug)
    val_ds = GapTokenDataset("val", vocab, pos_map, debug)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoader for the test set.
    """
    vocab, pos_map, _ = load_or_build_artifacts(load_cached_data=True)
    test_ds = GapTokenDataset("test", vocab, pos_map)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
