import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Token Maps
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_couples(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a numpy array of shape (L,) where arr[i] is the index of the partner
    of base i, or -1 if unpaired.
    """
    couples = np.full(len(structure), -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                couples[i] = j
                couples[j] = i
    return couples


def one_hot_encode(seq, token_map, num_classes):
    """
    One-hot encodes a sequence string based on the provided map.
    Returns a numpy array of shape (L, num_classes).
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in token_map:
            arr[i, token_map[char]] = 1.0
    return arr


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, targets):
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features: (107, 18)
        # partner_indices: (107,)
        # targets: (68, 5)

        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        pidx = torch.tensor(self.partner_indices[idx], dtype=torch.long)
        targ = torch.tensor(self.targets[idx], dtype=torch.float32)

        return feat, pidx, targ


def preprocess_data(config, mode="train", load_cached_data=True):
    """
    Loads raw data, generates features (including partner identity), and targets.
    Handles caching to .npz files.
    """
    # Determine paths based on mode
    if mode == "train":
        csv_path = config.train_metadata_path
        cache_path = config.train_cache_path
    elif mode == "val":
        csv_path = config.val_metadata_path
        cache_path = config.val_cache_path
    elif mode == "test":
        csv_path = config.test_metadata_path
        cache_path = config.test_cache_path
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            return data["features"], data["partner_indices"], data["targets"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Preprocessing {mode} data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Debugging: Limit size
    if config.sample_size is not None and mode in ["train", "val"]:
        df = df.head(config.sample_size)

    features_list = []
    partner_indices_list = []
    targets_list = []

    for _, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        length = len(seq)

        # Basic One-Hot Features
        oh_seq = one_hot_encode(seq, SEQ_MAP, 4)  # (L, 4)
        oh_struct = one_hot_encode(struct, STRUCT_MAP, 3)  # (L, 3)
        oh_loop = one_hot_encode(loop, LOOP_MAP, 7)  # (L, 7)

        # Partner Index Map
        # couples[i] = j means i is paired with j. couples[i] = -1 means unpaired.
        couples = get_couples(struct)

        # Partner Identity Feature
        # If i is paired with j, feature is one_hot(seq[j]).
        # If i is unpaired, feature is all zeros.
        oh_partner = np.zeros((length, 4), dtype=np.float32)
        for i, partner_idx in enumerate(couples):
            if partner_idx != -1:
                oh_partner[i] = oh_seq[partner_idx]

        # Concatenate all features: (L, 4+3+7+4) = (L, 18)
        sample_features = np.concatenate(
            [oh_seq, oh_struct, oh_loop, oh_partner], axis=1
        )

        features_list.append(sample_features)
        partner_indices_list.append(couples)

        # Process Targets
        if mode == "test":
            # Dummy targets for test set: (68, 5)
            # 5 columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            sample_targets = np.zeros((config.pred_len, 5), dtype=np.float32)
        else:
            # Parse stringified lists
            # We need to stack them in the order of config.target_cols
            t_arrays = []
            for col in config.target_cols:
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                    val_arr = np.array(val_list, dtype=np.float32)
                except:
                    # Fallback for malformed data (though metadata script should handle this)
                    val_arr = np.zeros(config.pred_len, dtype=np.float32)

                # Ensure length matches pred_len (68)
                if len(val_arr) < config.pred_len:
                    pad = np.zeros(config.pred_len - len(val_arr), dtype=np.float32)
                    val_arr = np.concatenate([val_arr, pad])
                elif len(val_arr) > config.pred_len:
                    val_arr = val_arr[: config.pred_len]

                t_arrays.append(val_arr)

            # Stack to shape (68, 5)
            sample_targets = np.stack(t_arrays, axis=1)

        targets_list.append(sample_targets)

    # Convert lists to numpy arrays
    features = np.array(features_list, dtype=np.float32)  # (N, 107, 18)
    partner_indices = np.array(partner_indices_list, dtype=np.int32)  # (N, 107)
    targets = np.array(targets_list, dtype=np.float32)  # (N, 68, 5)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path, features=features, partner_indices=partner_indices, targets=targets
    )
    print(f"Saved {mode} data to cache: {cache_path}")

    return features, partner_indices, targets


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    config = Config(debug=debug)

    # Load Data
    train_feats, train_pids, train_targs = preprocess_data(
        config, mode="train", load_cached_data=True
    )
    val_feats, val_pids, val_targs = preprocess_data(
        config, mode="val", load_cached_data=True
    )
    test_feats, test_pids, test_targs = preprocess_data(
        config, mode="test", load_cached_data=True
    )

    # Create Datasets
    train_dataset = RNADataset(train_feats, train_pids, train_targs)
    val_dataset = RNADataset(val_feats, val_pids, val_targs)
    test_dataset = RNADataset(test_feats, test_pids, test_targs)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
