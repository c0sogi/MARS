import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNAProcessor:
    """
    Handles feature engineering for RNA sequences, including:
    1. One-hot encoding of Sequence, Structure, and Predicted Loop Type.
    2. Parsing secondary structure to identify base pairs.
    3. Generating 'Partner Identity' features (one-hot of the paired base).
    4. Generating 'Partner Indices' for the gather operation in the model.
    """

    def __init__(self):
        # Dictionaries for One-Hot Encoding
        self.seq_map = {c: i for i, c in enumerate("AGCU")}
        self.struct_map = {c: i for i, c in enumerate(".()")}
        self.loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    def get_structure_pairs(self, structure):
        """
        Parses dot-bracket structure to find pairs.
        Returns a dictionary mapping index -> partner_index.
        """
        pairs = {}
        stack = []
        for i, char in enumerate(structure):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    start = stack.pop()
                    pairs[start] = i
                    pairs[i] = start
        return pairs

    def process_sample(self, row):
        """
        Generates features for a single sample row.
        """
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]
        length = len(sequence)

        # 1. Standard One-Hot Encodings
        # Sequence (4)
        seq_oh = np.zeros((length, 4), dtype=np.float32)
        for i, char in enumerate(sequence):
            if char in self.seq_map:
                seq_oh[i, self.seq_map[char]] = 1.0

        # Structure (3)
        struct_oh = np.zeros((length, 3), dtype=np.float32)
        for i, char in enumerate(structure):
            if char in self.struct_map:
                struct_oh[i, self.struct_map[char]] = 1.0

        # Loop Type (7)
        loop_oh = np.zeros((length, 7), dtype=np.float32)
        for i, char in enumerate(loop_type):
            if char in self.loop_map:
                loop_oh[i, self.loop_map[char]] = 1.0

        # 2. Partner Features
        pairs = self.get_structure_pairs(structure)

        # Partner Identity (4) - One-hot of the paired base
        partner_identity_oh = np.zeros((length, 4), dtype=np.float32)

        # Partner Indices (1) - Index of partner for gather, self if unpaired
        partner_indices = np.arange(length, dtype=np.int64)

        for i in range(length):
            if i in pairs:
                partner_idx = pairs[i]
                partner_indices[i] = partner_idx

                # Get partner base
                partner_base = sequence[partner_idx]
                if partner_base in self.seq_map:
                    partner_identity_oh[i, self.seq_map[partner_base]] = 1.0
            else:
                # Unpaired: partner_indices[i] remains i (self)
                # partner_identity_oh remains all zeros
                pass

        # Concatenate all features along channel dimension
        # Result shape: (Length, 18) -> Transpose later to (18, Length)
        features = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_identity_oh], axis=1
        )

        return features, partner_indices

    def process_dataframe(self, df):
        """
        Processes the entire dataframe.
        """
        all_features = []
        all_indices = []
        all_targets = []

        # Check if targets exist (Train/Val) or not (Test)
        has_targets = Config.TARGET_COLS[0] in df.columns

        for idx, row in df.iterrows():
            # Inputs
            feats, p_idxs = self.process_sample(row)
            all_features.append(feats)
            all_indices.append(p_idxs)

            # Targets
            if has_targets:
                # Targets are stored as stringified lists in the CSV
                # We need to parse them into arrays
                # Shape: (Seq_Len, 5)
                # Note: The CSV might have shorter lists (68) than seq length (107).
                # We pad with zeros to 107.

                sample_targets = np.zeros(
                    (Config.SEQ_LEN, len(Config.TARGET_COLS)), dtype=np.float32
                )

                for t_i, col in enumerate(Config.TARGET_COLS):
                    val_str = row[col]
                    try:
                        val_list = ast.literal_eval(val_str)
                        # Fill the available positions
                        length = min(len(val_list), Config.SEQ_LEN)
                        sample_targets[:length, t_i] = val_list[:length]
                    except (ValueError, SyntaxError):
                        # Handle cases where parsing fails (should not happen with clean metadata)
                        pass

                all_targets.append(sample_targets)

        # Stack into numpy arrays
        # Features: (N, L, C) -> Transpose to (N, C, L) for PyTorch Conv1d
        X = np.array(all_features, dtype=np.float32).transpose(0, 2, 1)
        I = np.array(all_indices, dtype=np.int64)

        if has_targets:
            Y = np.array(all_targets, dtype=np.float32)
        else:
            # Dummy targets for test set
            Y = np.zeros(
                (len(df), Config.SEQ_LEN, len(Config.TARGET_COLS)), dtype=np.float32
            )

        return X, I, Y


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, targets, ids=None):
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (C, L)
        x = torch.from_numpy(self.features[idx])
        # Partner Indices: (L,)
        p_idx = torch.from_numpy(self.partner_indices[idx])
        # Targets: (L, 5)
        y = torch.from_numpy(self.targets[idx])

        if self.ids is not None:
            return x, p_idx, y, self.ids[idx]
        return x, p_idx, y


def load_or_process_data(csv_path, cache_path, load_cached_data=True, debug=False):
    """
    Loads data from cache if available and requested.
    Otherwise, loads from CSV, processes features, and saves to cache.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["indices"], data["targets"], data["ids"]

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    processor = RNAProcessor()
    features, indices, targets = processor.process_dataframe(df)
    ids = df["id"].values

    # Save to cache
    print(f"Saving processed data to {cache_path}")
    np.savez_compressed(
        cache_path, features=features, indices=indices, targets=targets, ids=ids
    )

    return features, indices, targets, ids


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    set_seed(Config.SEED)

    # 1. Load Train Data
    train_X, train_I, train_Y, train_ids = load_or_process_data(
        Config.TRAIN_CSV,
        Config.TRAIN_CACHE,
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # 2. Load Val Data
    val_X, val_I, val_Y, val_ids = load_or_process_data(
        Config.VAL_CSV,
        Config.VAL_CACHE,
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # 3. Load Test Data
    test_X, test_I, test_Y, test_ids = load_or_process_data(
        Config.TEST_CSV,
        Config.TEST_CACHE,
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # Create Datasets
    train_dataset = RNADataset(train_X, train_I, train_Y, train_ids)
    val_dataset = RNADataset(val_X, val_I, val_Y, val_ids)
    test_dataset = RNADataset(test_X, test_I, test_Y, test_ids)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
