import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, pair_masks, targets=None, ids=None):
        """
        PyTorch Dataset for RNA Degradation Prediction.

        Args:
            inputs: (N, 107, 14) float32 array. One-hot encoded features.
            pair_indices: (N, 107) int32 array. Indices of paired bases.
            pair_masks: (N, 107) float32 array. 1.0 if paired, 0.0 otherwise.
            targets: (N, 68, 5) float32 array. Ground truth values (optional).
            ids: (N,) array of sample IDs (optional).
        """
        self.inputs = inputs
        self.pair_indices = pair_indices
        self.pair_masks = pair_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert inputs to tensors
        inp = torch.from_numpy(self.inputs[idx]).float()
        p_idx = torch.from_numpy(self.pair_indices[idx]).long()
        p_mask = torch.from_numpy(self.pair_masks[idx]).float()

        sample = {"inputs": inp, "pair_indices": p_idx, "pair_masks": p_mask}

        # Add targets if available (Train/Val)
        if self.targets is not None:
            sample["targets"] = torch.from_numpy(self.targets[idx]).float()

        # Add ID if available
        if self.ids is not None:
            sample["ids"] = str(self.ids[idx])

        return sample


def parse_structure(structure_str, seq_len):
    """
    Parses dot-bracket structure string to generate adjacency maps.

    Returns:
        pair_index: (seq_len,) int32. Maps index i to its partner j.
                    Unpaired positions map to 0 (safe dummy).
        pair_mask: (seq_len,) float32. 1.0 if paired, 0.0 if unpaired.
    """
    pair_index = np.zeros(seq_len, dtype=np.int32)
    pair_mask = np.zeros(seq_len, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if i >= seq_len:
            break

        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Set bidirectional mapping
                pair_index[i] = j
                pair_index[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0

    return pair_index, pair_mask


def encode_sequence(seq_str, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq_str):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_dataframe(df, mode):
    """
    Converts a pandas DataFrame into numpy arrays for the dataset.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_channels = Config.INPUT_CHANNELS

    # Pre-allocate arrays for efficiency
    inputs = np.zeros((num_samples, seq_len, input_channels), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Allocate targets only for training/validation
    targets = None
    if mode in ["train", "val"]:
        pred_len = Config.PRED_LEN
        num_targets = Config.NUM_TARGETS
        targets = np.zeros((num_samples, pred_len, num_targets), dtype=np.float32)

    # Iterate over rows
    # Note: iterrows is used here for clarity and because dataset size (~2k) is small enough.
    for idx, row in df.iterrows():
        # 1. Feature Engineering (Strict One-Hot)
        # Sequence (4 channels)
        seq_feat = encode_sequence(row["sequence"], Config.TOKEN_MAP_SEQ, seq_len)
        # Structure (3 channels)
        struct_feat = encode_sequence(
            row["structure"], Config.TOKEN_MAP_STRUCT, seq_len
        )
        # Loop Type (7 channels)
        loop_feat = encode_sequence(
            row["predicted_loop_type"], Config.TOKEN_MAP_LOOP, seq_len
        )

        # Concatenate features: (107, 14)
        inputs[idx] = np.concatenate([seq_feat, struct_feat, loop_feat], axis=1)

        # 2. Adjacency Map Construction
        p_idx, p_mask = parse_structure(row["structure"], seq_len)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Target Extraction
        if mode in ["train", "val"]:
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Parquet preserves lists. Ensure we only take the scored length (68).
                if isinstance(val_list, (list, np.ndarray)):
                    length = min(len(val_list), pred_len)
                    targets[idx, :length, t_i] = val_list[:length]

    return inputs, pair_indices, pair_masks, targets, ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.
    Handles caching to speed up subsequent runs.
    """
    # Map modes to cache file paths (using .npz for multiple arrays)
    cache_files = {
        "train": Config.TRAIN_CACHE.replace(".npy", ".npz"),
        "val": Config.VAL_CACHE.replace(".npy", ".npz"),
        "test": Config.TEST_CACHE.replace(".npy", ".npz"),
    }

    # Map modes to source metadata files
    metadata_files = {
        "train": Config.TRAIN_METADATA,
        "val": Config.VAL_METADATA,
        "test": Config.TEST_METADATA,
    }

    datasets = {}

    for mode in ["train", "val", "test"]:
        cache_path = cache_files[mode]
        loaded = False

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                inputs = data["inputs"]
                pair_indices = data["pair_indices"]
                pair_masks = data["pair_masks"]
                ids = data["ids"]

                # Load targets if they exist (Train/Val)
                targets = None
                if "targets" in data:
                    targets = data["targets"]
                    # np.load might return a 0-d array if None was saved, handle carefully
                    if targets.ndim == 0:
                        targets = None

                print(f"[{mode.upper()}] Loaded data from cache: {cache_path}")
                loaded = True
            except Exception as e:
                print(f"[{mode.upper()}] Failed to load cache: {e}")
                loaded = False

        # 2. Process from source if cache failed or disabled
        if not loaded:
            print(
                f"[{mode.upper()}] Processing data from metadata: {metadata_files[mode]}"
            )
            df = pd.read_parquet(metadata_files[mode])
            inputs, pair_indices, pair_masks, targets, ids = process_dataframe(df, mode)

            # Save to cache
            save_dict = {
                "inputs": inputs,
                "pair_indices": pair_indices,
                "pair_masks": pair_masks,
                "ids": ids,
            }
            if targets is not None:
                save_dict["targets"] = targets

            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, **save_dict)
            print(f"[{mode.upper()}] Saved cache to {cache_path}")

        # 3. Instantiate Dataset
        datasets[mode] = RNADataset(inputs, pair_indices, pair_masks, targets, ids)

    # 4. Create DataLoaders
    # Pin memory for faster transfer to GPU
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batch for stability in training
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
