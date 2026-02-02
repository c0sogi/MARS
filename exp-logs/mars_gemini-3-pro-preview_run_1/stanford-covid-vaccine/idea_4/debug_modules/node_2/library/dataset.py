import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Holds pre-processed integer-encoded sequences and targets.
    """

    def __init__(self, ids, sequences, structures, loops, targets=None, masks=None):
        self.ids = ids
        self.sequences = sequences
        self.structures = structures
        self.loops = loops
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        item = {
            "id": self.ids[idx],
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.long),
            "structure": torch.tensor(self.structures[idx], dtype=torch.long),
            "predicted_loop_type": torch.tensor(self.loops[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.masks is not None:
            item["mask"] = torch.tensor(self.masks[idx], dtype=torch.float32)

        return item


def encode_sequence(seq_str, vocab, max_len):
    """Encodes a string sequence into a fixed-length integer array."""
    # Map characters to integers
    tokenized = [vocab.get(char, -1) for char in seq_str]
    # Pad or truncate (though data is fixed length 107)
    if len(tokenized) < max_len:
        tokenized += [-1] * (max_len - len(tokenized))
    else:
        tokenized = tokenized[:max_len]
    return np.array(tokenized, dtype=np.int16)


def preprocess_dataframe(df, mode="train"):
    """
    Converts a pandas DataFrame into numpy arrays for the dataset.
    Handles tokenization and target padding.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    ids = df["id"].values
    sequences = np.zeros((num_samples, seq_len), dtype=np.int16)
    structures = np.zeros((num_samples, seq_len), dtype=np.int16)
    loops = np.zeros((num_samples, seq_len), dtype=np.int16)

    # Targets and masks are only for train/val
    targets = None
    masks = None

    if mode in ["train", "val"]:
        # Shape: (N, 107, 5)
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
        # Shape: (N, 107) - Binary mask (1 for scored, 0 for unscored)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Process each row
    # Note: Vectorization is possible but loop is clearer for complex parsing and safe given dataset size (~2k)
    for i, row in df.iterrows():
        # 1. Inputs
        sequences[i] = encode_sequence(row["sequence"], Config.VOCAB_SEQ, seq_len)
        structures[i] = encode_sequence(row["structure"], Config.VOCAB_STRUCT, seq_len)
        loops[i] = encode_sequence(
            row["predicted_loop_type"], Config.VOCAB_LOOP, seq_len
        )

        # 2. Targets (only for train/val)
        if mode in ["train", "val"]:
            seq_scored = row["seq_scored"]

            # Create mask: 1 for first 'seq_scored' positions, 0 otherwise
            masks[i, :seq_scored] = 1.0

            # Extract targets
            # Each target column in parquet is a list/array of length seq_scored (68)
            for t_idx, col_name in enumerate(Config.TARGET_COLS):
                val_list = row[col_name]
                # Assign to the corresponding slice in the target tensor
                # We trust the metadata generation that lengths match seq_scored
                length = len(val_list)
                targets[i, :length, t_idx] = val_list

    return ids, sequences, structures, loops, targets, masks


def load_or_process_data(metadata_path, cache_name, mode, load_cached_data):
    """
    Caching logic:
    1. Check if cache exists.
    2. If exists and load_cached_data=True, load from .npz.
    3. Else, load parquet, preprocess, save to .npz, return arrays.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            ids = data["ids"]
            sequences = data["sequences"]
            structures = data["structures"]
            loops = data["loops"]

            if mode in ["train", "val"]:
                targets = data["targets"]
                masks = data["masks"]
                return ids, sequences, structures, loops, targets, masks
            else:
                return ids, sequences, structures, loops, None, None
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    # Handle Debugging
    if Config.DEBUG:
        print(
            f"DEBUG mode: limiting {mode} data to {Config.DEBUG_SUBSET_SIZE} samples."
        )
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy().reset_index(drop=True)

    ids, sequences, structures, loops, targets, masks = preprocess_dataframe(df, mode)

    # Save to cache
    print(f"Saving processed data to {cache_file}...")
    save_dict = {
        "ids": ids,
        "sequences": sequences,
        "structures": structures,
        "loops": loops,
    }
    if targets is not None:
        save_dict["targets"] = targets
        save_dict["masks"] = masks

    np.savez_compressed(cache_file, **save_dict)

    return ids, sequences, structures, loops, targets, masks


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching, dataset creation, and loader instantiation.
    """
    seed_everything(Config.SEED)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # --- Train Data ---
    train_ids, train_seq, train_struct, train_loop, train_tgt, train_mask = (
        load_or_process_data(
            Config.TRAIN_METADATA, "train_data", "train", load_cached_data
        )
    )
    train_dataset = RNADataset(
        train_ids, train_seq, train_struct, train_loop, train_tgt, train_mask
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Validation Data ---
    val_ids, val_seq, val_struct, val_loop, val_tgt, val_mask = load_or_process_data(
        Config.VAL_METADATA, "val_data", "val", load_cached_data
    )
    val_dataset = RNADataset(val_ids, val_seq, val_struct, val_loop, val_tgt, val_mask)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test Data ---
    test_ids, test_seq, test_struct, test_loop, _, _ = load_or_process_data(
        Config.TEST_METADATA, "test_data", "test", load_cached_data
    )
    test_dataset = RNADataset(
        test_ids, test_seq, test_struct, test_loop, targets=None, masks=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
