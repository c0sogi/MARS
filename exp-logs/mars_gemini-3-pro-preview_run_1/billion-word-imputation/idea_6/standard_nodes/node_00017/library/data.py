import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.vocab import Vocabulary


def prepare_data(split, vocab, load_cached_data=True):
    """
    Prepares data for the given split.
    Loads from cache if available and requested.
    Otherwise, loads from metadata CSV, tokenizes, and saves to cache.

    Args:
        split (str): 'train', 'val', or 'test'.
        vocab (Vocabulary): Vocabulary instance for tokenization.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'token_ids'.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_tokens.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}. Rebuilding data.")

    # 2. Determine source file
    if split == "train":
        csv_path = Config.TRAIN_METADATA
    elif split == "val":
        csv_path = Config.VAL_METADATA
    elif split == "test":
        csv_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    # 3. Load and Process Data
    df = pd.read_csv(csv_path)

    # Pre-fetch vocabulary mapping for speed
    stoi = vocab.stoi
    unk_id = stoi[Config.UNK_TOKEN]

    def tokenize_row(text):
        if not isinstance(text, str):
            return []
        return [stoi.get(t, unk_id) for t in text.split()]

    # Tokenize all sentences
    sentences = df["sentence"].astype(str).tolist()
    token_ids_list = [tokenize_row(s) for s in sentences]

    # 4. Filter Data
    # For train/val, we need at least 3 tokens to remove a middle word (not first/last).
    # Test set must preserve all rows.
    valid_indices = []
    if split in ["train", "val"]:
        for i, tokens in enumerate(token_ids_list):
            if len(tokens) >= 3:
                valid_indices.append(i)
    else:
        valid_indices = range(len(token_ids_list))

    # Create filtered DataFrame
    filtered_ids = df.iloc[valid_indices]["id"].values
    filtered_tokens = [token_ids_list[i] for i in valid_indices]

    processed_df = pd.DataFrame({"id": filtered_ids, "token_ids": filtered_tokens})

    # 5. Save to cache
    processed_df.to_parquet(cache_path, index=False)

    return processed_df


class InterleavedDataset(Dataset):
    def __init__(self, split, vocab, load_cached_data=True, max_samples=None):
        """
        Dataset for Bifurcated Interleaved Transformer.

        Args:
            split (str): 'train', 'val', or 'test'.
            vocab (Vocabulary): Vocabulary instance.
            load_cached_data (bool): Whether to use cached parquet files.
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.vocab = vocab
        self.gap_id = vocab.stoi[Config.GAP_TOKEN]

        # Load data
        self.data = prepare_data(split, vocab, load_cached_data)

        if max_samples is not None:
            self.data = self.data.iloc[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_ids = row["token_ids"]  # List[int]
        sentence_id = row["id"]

        # Ensure list type (parquet might load as numpy array)
        if isinstance(original_ids, np.ndarray):
            original_ids = original_ids.tolist()

        if self.split in ["train", "val"]:
            # --- Training Logic: Dynamic Masking ---
            # Rule: Remove one word, uniformly random, never first or last.
            seq_len = len(original_ids)

            # Pick index to remove: range [1, seq_len - 2]
            # e.g. A B C (len 3). Remove B (idx 1). randint(1, 2) -> 1.
            remove_idx = np.random.randint(1, seq_len - 1)
            target_word_id = original_ids[remove_idx]

            # Create remaining sequence by removing the word
            remaining_ids = original_ids[:remove_idx] + original_ids[remove_idx + 1 :]

            # Interleave with GAPs: [w0, GAP, w1, GAP, w2, ...]
            interleaved_ids = []
            for i, tid in enumerate(remaining_ids):
                interleaved_ids.append(tid)
                if i < len(remaining_ids) - 1:
                    interleaved_ids.append(self.gap_id)

            # Calculate Target Gap Index
            # The missing word was between `remove_idx-1` and `remove_idx+1` in original.
            # In `remaining_ids`, the word preceding the gap is at `remove_idx-1`.
            # In `interleaved_ids`, word at index `j` is at position `2*j`.
            # The gap following word `j` is at position `2*j + 1`.
            # Therefore, target gap index = 2 * (remove_idx - 1) + 1 = 2 * remove_idx - 1.
            target_gap_idx = 2 * remove_idx - 1

            # Truncation
            if len(interleaved_ids) > Config.MAX_SEQ_LEN:
                interleaved_ids = interleaved_ids[: Config.MAX_SEQ_LEN]
                # If target gap is truncated, ignore this sample
                if target_gap_idx >= Config.MAX_SEQ_LEN:
                    target_gap_idx = -1
                    target_word_id = -1

            return {
                "input_ids": torch.tensor(interleaved_ids, dtype=torch.long),
                "target_loc": torch.tensor(target_gap_idx, dtype=torch.long),
                "target_word": torch.tensor(target_word_id, dtype=torch.long),
                "id": sentence_id,
            }

        else:
            # --- Test Logic: Just Interleave ---
            # Word is already removed.
            interleaved_ids = []
            for i, tid in enumerate(original_ids):
                interleaved_ids.append(tid)
                if i < len(original_ids) - 1:
                    interleaved_ids.append(self.gap_id)

            # Truncation
            if len(interleaved_ids) > Config.MAX_SEQ_LEN:
                interleaved_ids = interleaved_ids[: Config.MAX_SEQ_LEN]

            return {
                "input_ids": torch.tensor(interleaved_ids, dtype=torch.long),
                "target_loc": torch.tensor(-1, dtype=torch.long),  # Dummy
                "target_word": torch.tensor(-1, dtype=torch.long),  # Dummy
                "id": sentence_id,
            }


def collate_fn(batch):
    """
    Collate function to pad sequences and stack tensors.
    """
    input_ids = [item["input_ids"] for item in batch]
    target_locs = [item["target_loc"] for item in batch]
    target_words = [item["target_word"] for item in batch]
    ids = [item["id"] for item in batch]

    # Pad inputs
    # PAD token is index 0 in Vocabulary
    pad_id = 0
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)

    # Stack targets
    target_locs = torch.stack(target_locs)
    target_words = torch.stack(target_words)

    # Create Attention Mask (1 for real tokens, 0 for pad)
    attention_mask = (input_ids_padded != pad_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "target_loc": target_locs,
        "target_word": target_words,
        "id": ids,
    }
