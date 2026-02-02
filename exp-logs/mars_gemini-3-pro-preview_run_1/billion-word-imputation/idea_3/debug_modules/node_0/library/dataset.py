import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import tokenize, SOS_TOKEN, EOS_TOKEN


class GapDataset(Dataset):
    def __init__(self, metadata_path, vocab, mode="train", load_cached_data=True):
        """
        Dataset class for the Gap Insertion Task.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            vocab (Vocabulary): Vocabulary object for encoding.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.
        """
        self.metadata_path = metadata_path
        self.vocab = vocab
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Determine cache filename
        # We append _debug if in debug mode to avoid overwriting full cache with partial data
        debug_suffix = "_debug" if Config.DEBUG else ""
        self.cache_path = os.path.join(
            Config.WORK_DIR, f"{mode}_tokens{debug_suffix}.parquet"
        )

        self.data = self._load_or_create_data()

        # Get PAD token ID for padding in collate (assuming it is 0 based on Utils)
        self.pad_idx = self.vocab.stoi.get(Config.PAD_TOKEN, 0)

    def _load_or_create_data(self):
        # 1. Try Loading Cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            print(f"[{self.mode.upper()}] Loading cached data from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                print(f"[{self.mode.upper()}] Loaded {len(df)} samples.")
                return df
            except Exception as e:
                print(f"[{self.mode.upper()}] Failed to load cache: {e}. Recreating...")

        # 2. Create from Scratch
        print(f"[{self.mode.upper()}] Processing data from {self.metadata_path}...")

        # Load CSV
        nrows = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None

        try:
            df = pd.read_csv(self.metadata_path, nrows=nrows)
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return pd.DataFrame()  # Return empty on failure

        # Tokenization and Encoding Helper
        def process_text(text):
            # Tokenize
            tokens = tokenize(str(text))
            # Add SOS and EOS
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            # Encode
            return self.vocab.encode(tokens)

        # Apply processing
        # Using list comprehension for speed
        if "sentence" in df.columns:
            sentences = df["sentence"].fillna("").tolist()
            encoded_data = [process_text(s) for s in sentences]
        else:
            encoded_data = []

        # Create DataFrame for storage
        out_df = pd.DataFrame({"id": df["id"], "token_ids": encoded_data})

        # 3. Save to Cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        out_df.to_parquet(self.cache_path, index=False)
        print(f"[{self.mode.upper()}] Saved processed data to {self.cache_path}")

        return out_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        token_ids = row["token_ids"]  # List of ints

        # Test Mode: Return sequence and ID
        if self.mode == "test":
            return torch.tensor(token_ids, dtype=torch.long), row["id"]

        # Train/Val Mode: Dynamic Masking
        # Sequence: [SOS, w1, w2, ..., wn, ., EOS]
        L = len(token_ids)

        # Valid removal range: Indices [2, L-3]
        # Corresponds to removing words between first word and last word (period)
        # Minimum length required: 5 (SOS, First, Middle, Period, EOS) -> Remove Middle

        if L < 5:
            # Fallback for very short sentences (rare/edge case)
            # Remove index 1 (First word)
            remove_idx = 1
        else:
            if self.mode == "val":
                # Deterministic randomness for validation
                rng = np.random.RandomState(idx)
                # randint(low, high) -> [low, high)
                # We want max index L-3. So high = L-2.
                remove_idx = rng.randint(2, L - 2)
            else:
                remove_idx = np.random.randint(2, L - 2)

        # Ensure remove_idx is within bounds (safety)
        remove_idx = min(max(1, remove_idx), L - 2)

        target_token_id = token_ids[remove_idx]

        # Create input: remove token at remove_idx
        input_ids = token_ids[:remove_idx] + token_ids[remove_idx + 1 :]

        # Gap Index: The gap is after the token that was at remove_idx - 1
        gap_idx = remove_idx - 1

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(gap_idx, dtype=torch.long),
            torch.tensor(target_token_id, dtype=torch.long),
        )


def collate_fn(batch):
    """
    Collate function to pad sequences and stack targets.
    """
    elem_len = len(batch[0])

    # Check if Train/Val (3 elements) or Test (2 elements)
    if elem_len == 3:
        input_ids_list, gap_idxs, target_ids = zip(*batch)

        # Pad inputs (batch_first=True)
        # We assume PAD token index is 0.
        padded_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=0
        )

        gap_idxs = torch.stack(gap_idxs)
        target_ids = torch.stack(target_ids)

        # Attention Mask (1 for real tokens, 0 for pad)
        attention_mask = (padded_input_ids != 0).long()

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "gap_idx": gap_idxs,
            "target_id": target_ids,
        }
    else:
        input_ids_list, row_ids = zip(*batch)

        padded_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=0
        )
        attention_mask = (padded_input_ids != 0).long()

        # row_ids might be int or string, usually int in this dataset
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "row_id": torch.tensor(row_ids, dtype=torch.long),
        }


def get_dataloaders(vocab, load_cached_data=True):
    """
    Factory function to create DataLoaders.
    """
    # Train
    train_ds = GapDataset(
        Config.TRAIN_METADATA, vocab, mode="train", load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    # Val
    val_ds = GapDataset(
        Config.VAL_METADATA, vocab, mode="val", load_cached_data=load_cached_data
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    # Test
    test_ds = GapDataset(
        Config.TEST_METADATA, vocab, mode="test", load_cached_data=load_cached_data
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
