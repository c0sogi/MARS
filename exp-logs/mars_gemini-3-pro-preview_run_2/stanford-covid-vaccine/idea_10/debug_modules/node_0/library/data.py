import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SEQ_LEN,
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_STRUCT,
    VOCAB_SIZE_LOOP,
    CACHE_VERSION,
    BATCH_SIZE,
    ALL_TARGETS,
    SCORED_TARGETS,
)
from library.utils import parse_list_column

# =============================================================================
# MAPPINGS
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_indices(structure):
    """
    Parses a dot-bracket structure string to find paired indices.
    Returns a numpy array of shape (seq_len,) where arr[i] is the index of the
    partner of base i. If i is unpaired, arr[i] is -1.
    """
    partners = np.full(len(structure), -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i

    return partners


def encode_sequence(seq, mapping):
    return np.array([mapping.get(c, 0) for c in seq], dtype=np.int32)


# =============================================================================
# DATASET CLASS
# =============================================================================
class RNADataset(Dataset):
    def __init__(
        self,
        seq_indices,
        struct_indices,
        loop_indices,
        partner_indices,
        targets=None,
        ids=None,
    ):
        """
        Args:
            seq_indices: (N, SeqLen)
            struct_indices: (N, SeqLen)
            loop_indices: (N, SeqLen)
            partner_indices: (N, SeqLen)
            targets: (N, SeqLen, NumTargets) or None
            ids: List of IDs
        """
        self.seq_indices = seq_indices
        self.struct_indices = struct_indices
        self.loop_indices = loop_indices
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.seq_indices)

    def __getitem__(self, idx):
        # 1. Retrieve Indices
        s_idx = self.seq_indices[idx]  # (L,)
        st_idx = self.struct_indices[idx]  # (L,)
        l_idx = self.loop_indices[idx]  # (L,)
        p_idx = self.partner_indices[idx]  # (L,)

        # 2. One-Hot Encoding
        # We convert indices to one-hot floats
        # Sequence
        oh_seq = np.eye(VOCAB_SIZE_SEQ, dtype=np.float32)[s_idx]  # (L, 4)
        # Structure
        oh_struct = np.eye(VOCAB_SIZE_STRUCT, dtype=np.float32)[st_idx]  # (L, 3)
        # Loop
        oh_loop = np.eye(VOCAB_SIZE_LOOP, dtype=np.float32)[l_idx]  # (L, 7)

        # 3. Concatenate Features
        # Shape: (L, Channels) -> (L, 14)
        features = np.concatenate([oh_seq, oh_struct, oh_loop], axis=1)

        # Transpose to (Channels, L) for Conv1d compatibility
        features = features.transpose(1, 0)  # (14, L)

        # 4. Prepare Tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        partner_tensor = torch.tensor(p_idx, dtype=torch.long)

        # 5. Handle Targets
        if self.targets is not None:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            return features_tensor, partner_tensor, target_tensor
        else:
            # For inference, return ID as well if needed, but usually DataLoader order is preserved.
            # We return a dummy target for consistency or just the inputs.
            # Returning features, partners, and the ID for submission mapping.
            id_val = self.ids[idx]
            return features_tensor, partner_tensor, id_val


# =============================================================================
# PROCESSING & LOADING
# =============================================================================
def process_data(csv_path, mode="train"):
    """
    Reads CSV, parses sequences/structures/targets, and returns numpy arrays.
    """
    df = pd.read_csv(csv_path)

    # Pre-allocate arrays
    num_samples = len(df)

    seq_arr = np.zeros((num_samples, SEQ_LEN), dtype=np.int32)
    struct_arr = np.zeros((num_samples, SEQ_LEN), dtype=np.int32)
    loop_arr = np.zeros((num_samples, SEQ_LEN), dtype=np.int32)
    partner_arr = np.zeros((num_samples, SEQ_LEN), dtype=np.int32)

    # Targets: (N, 107, 5)
    # We initialize with zeros (padding)
    target_arr = None
    if mode != "test":
        target_arr = np.zeros(
            (num_samples, SEQ_LEN, len(ALL_TARGETS)), dtype=np.float32
        )

    ids = df["id"].tolist()

    for i, row in df.iterrows():
        # Inputs
        seq_arr[i] = encode_sequence(row["sequence"], SEQ_MAP)
        struct_arr[i] = encode_sequence(row["structure"], STRUCT_MAP)
        loop_arr[i] = encode_sequence(row["predicted_loop_type"], LOOP_MAP)
        partner_arr[i] = get_partner_indices(row["structure"])

        # Targets
        if mode != "test":
            # Parse each target column
            # Note: Targets in CSV are length 68 (seq_scored). We pad to 107.
            for t_idx, col_name in enumerate(ALL_TARGETS):
                val = parse_list_column(row[col_name])
                length = len(val)
                if length > 0:
                    # Fill the first 'length' positions
                    target_arr[i, :length, t_idx] = val

    return seq_arr, struct_arr, loop_arr, partner_arr, target_arr, ids


def get_loaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching logic.
    """

    splits = ["train", "val", "test"]
    datasets = {}

    for split in splits:
        cache_path = os.path.join(WORKING_DIR, f"{split}_data_{CACHE_VERSION}.npz")
        csv_path = os.path.join(METADATA_DIR, f"{split}.csv")

        data_loaded = False

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # print(f"Loading {split} data from cache: {cache_path}")
                loaded = np.load(cache_path, allow_pickle=True)
                seq_arr = loaded["seq_arr"]
                struct_arr = loaded["struct_arr"]
                loop_arr = loaded["loop_arr"]
                partner_arr = loaded["partner_arr"]
                ids = loaded["ids"].tolist()

                if split != "test":
                    target_arr = loaded["target_arr"]
                else:
                    target_arr = None

                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache for {split}: {e}. Reprocessing...")

        # 2. Process from Scratch if needed
        if not data_loaded:
            # print(f"Processing {split} data from {csv_path}...")
            if split == "test":
                seq_arr, struct_arr, loop_arr, partner_arr, target_arr, ids = (
                    process_data(csv_path, mode="test")
                )
                np.savez_compressed(
                    cache_path,
                    seq_arr=seq_arr,
                    struct_arr=struct_arr,
                    loop_arr=loop_arr,
                    partner_arr=partner_arr,
                    ids=ids,
                )
            else:
                seq_arr, struct_arr, loop_arr, partner_arr, target_arr, ids = (
                    process_data(csv_path, mode="train")
                )
                np.savez_compressed(
                    cache_path,
                    seq_arr=seq_arr,
                    struct_arr=struct_arr,
                    loop_arr=loop_arr,
                    partner_arr=partner_arr,
                    target_arr=target_arr,
                    ids=ids,
                )

        # 3. Create Dataset
        datasets[split] = RNADataset(
            seq_indices=seq_arr,
            struct_indices=struct_arr,
            loop_indices=loop_arr,
            partner_indices=partner_arr,
            targets=target_arr,
            ids=ids,
        )

    # 4. Create DataLoaders
    # Use a generator for reproducibility
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
