import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_list_column

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {c: i for i, c in enumerate(Config.BASES)}
STRUCT_MAP = {c: i for i, c in enumerate(Config.STRUCTURES)}
LOOP_MAP = {c: i for i, c in enumerate(Config.LOOP_TYPES)}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA data.
    Returns:
        features: (107, 18) - Concatenated One-Hot encodings + Partner Identity
        partner_indices: (107,) - Indices of paired bases (-1 if unpaired)
        targets: (107, 5) - Ground truth values (padded with 0 for unscored positions)
        sample_id: str - The ID of the sample
    """

    def __init__(self, features, partner_indices, targets, ids):
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to tensors
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample_id = self.ids[idx]

        return feat, p_idx, tgt, sample_id


def get_partners(structure):
    """
    Parses dot-bracket structure to find pairing partners.
    Returns an array of indices where arr[i] = j if i pairs with j, else -1.
    """
    partners = np.full(len(structure), -1, dtype=int)
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


def one_hot(indices, depth):
    """
    Creates a one-hot encoding numpy array.
    Args:
        indices: (L,) array of integers
        depth: int, number of classes
    Returns:
        (L, depth) float32 array
    """
    L = len(indices)
    out = np.zeros((L, depth), dtype=np.float32)

    # Create mask for valid indices (ignore -1 or out of bounds)
    valid = (indices >= 0) & (indices < depth)

    # Assign 1.0 to valid positions
    out[np.arange(L)[valid], indices[valid]] = 1.0
    return out


def preprocess_dataframe(df, is_test=False):
    """
    Processes a pandas DataFrame into numpy arrays for features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Feature dimensions
    dim_seq = len(Config.BASES)  # 4
    dim_struct = len(Config.STRUCTURES)  # 3
    dim_loop = len(Config.LOOP_TYPES)  # 7
    dim_partner = dim_seq  # 4 (Identity of the partner base)

    total_features = dim_seq + dim_struct + dim_loop + dim_partner

    # Pre-allocate arrays
    all_features = np.zeros((num_samples, seq_len, total_features), dtype=np.float32)
    all_partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    all_targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    all_ids = df["id"].values

    # Pre-parse target columns to lists of arrays to speed up the loop
    # We use the helper from utils.py
    parsed_targets = {}
    if not is_test:
        for col in Config.TARGET_COLS:
            parsed_targets[col] = df[col].apply(parse_list_column).values

    for i in range(num_samples):
        row = df.iloc[i]
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Sequence One-Hot
        seq_indices = np.array([SEQ_MAP.get(c, -1) for c in sequence])
        seq_oh = one_hot(seq_indices, dim_seq)

        # 2. Structure One-Hot
        struct_indices = np.array([STRUCT_MAP.get(c, -1) for c in structure])
        struct_oh = one_hot(struct_indices, dim_struct)

        # 3. Loop One-Hot
        loop_indices = np.array([LOOP_MAP.get(c, -1) for c in loop_type])
        loop_oh = one_hot(loop_indices, dim_loop)

        # 4. Partner Indices
        partners = get_partners(structure)
        all_partner_indices[i] = partners

        # 5. Partner Identity Feature
        # If base i is paired with j, feature at i includes sequence[j]
        partner_identity = np.zeros((seq_len, dim_partner), dtype=np.float32)
        valid_pairs = partners != -1

        if np.any(valid_pairs):
            indices_with_partners = np.where(valid_pairs)[0]
            partner_locs = partners[indices_with_partners]
            # Copy the sequence one-hot of the partner
            partner_identity[indices_with_partners] = seq_oh[partner_locs]

        # Concatenate all features
        # Shape: (107, 4+3+7+4) = (107, 18)
        sample_feat = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_identity], axis=1
        )
        all_features[i] = sample_feat

        # 6. Targets
        # Fill the first 68 positions with ground truth, rest remains 0
        if not is_test:
            for col_idx, col in enumerate(Config.TARGET_COLS):
                val_arr = parsed_targets[col][i]
                length = len(val_arr)
                if length > 0:
                    valid_len = min(length, seq_len)
                    all_targets[i, :valid_len, col_idx] = val_arr[:valid_len]

    return all_features, all_partner_indices, all_targets, all_ids


def load_data(load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from metadata CSVs.
    """
    cache_path = Config.CACHE_FILE

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["train_feat"],
                data["train_pidx"],
                data["train_tgt"],
                data["train_ids"],
                data["val_feat"],
                data["val_pidx"],
                data["val_tgt"],
                data["val_ids"],
                data["test_feat"],
                data["test_pidx"],
                data["test_tgt"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Process each split
    print("Processing Train...")
    train_feat, train_pidx, train_tgt, train_ids = preprocess_dataframe(
        train_df, is_test=False
    )

    print("Processing Validation...")
    val_feat, val_pidx, val_tgt, val_ids = preprocess_dataframe(val_df, is_test=False)

    print("Processing Test...")
    test_feat, test_pidx, test_tgt, test_ids = preprocess_dataframe(
        test_df, is_test=True
    )

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_feat=train_feat,
        train_pidx=train_pidx,
        train_tgt=train_tgt,
        train_ids=train_ids,
        val_feat=val_feat,
        val_pidx=val_pidx,
        val_tgt=val_tgt,
        val_ids=val_ids,
        test_feat=test_feat,
        test_pidx=test_pidx,
        test_tgt=test_tgt,
        test_ids=test_ids,
    )
    print(f"Data saved to {cache_path}")

    return (
        train_feat,
        train_pidx,
        train_tgt,
        train_ids,
        val_feat,
        val_pidx,
        val_tgt,
        val_ids,
        test_feat,
        test_pidx,
        test_tgt,
        test_ids,
    )


def get_loaders(load_cached_data=True):
    """
    Returns DataLoaders for train, validation, and test sets.
    """
    data = load_data(load_cached_data)
    (
        train_feat,
        train_pidx,
        train_tgt,
        train_ids,
        val_feat,
        val_pidx,
        val_tgt,
        val_ids,
        test_feat,
        test_pidx,
        test_tgt,
        test_ids,
    ) = data

    # Create Datasets
    train_ds = RNADataset(train_feat, train_pidx, train_tgt, train_ids)
    val_ds = RNADataset(val_feat, val_pidx, val_tgt, val_ids)
    test_ds = RNADataset(test_feat, test_pidx, test_tgt, test_ids)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
