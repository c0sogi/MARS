import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_list_column


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA sequences.
    Returns inputs, partner indices for structural gathering, and targets.
    """

    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs shape: (Channels, Seq_Len)
        # partner_indices shape: (Seq_Len,)

        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
        }

        if self.targets is not None:
            # targets shape: (5, Seq_Len)
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


class Preloader:
    """
    Handles loading, preprocessing, and caching of RNA data.
    """

    def __init__(self):
        # Mappings for One-Hot Encoding
        self.seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
        self.struct_map = {".": 0, "(": 1, ")": 2}
        self.loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    def get_partner_indices(self, structure):
        """
        Generates an index map for paired bases.
        If base i is paired with j, map[i] = j.
        If base i is unpaired, map[i] = i (self-reference).
        """
        length = len(structure)
        partners = np.arange(length)  # Default to self
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

    def one_hot(self, seq, map_dict, num_classes):
        """
        Converts a sequence string to a one-hot encoded numpy array (Channels, Length).
        """
        L = len(seq)
        arr = np.zeros((num_classes, L), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in map_dict:
                arr[map_dict[char], i] = 1.0
        return arr

    def process_df(self, df, is_test=False):
        """
        Iterates over the dataframe and generates feature arrays.
        """
        n_samples = len(df)
        seq_len = Config.SEQ_LEN
        in_channels = Config.IN_CHANNELS  # 14

        # Initialize arrays
        inputs = np.zeros((n_samples, in_channels, seq_len), dtype=np.float32)
        partner_indices = np.zeros((n_samples, seq_len), dtype=np.int32)
        ids = df["id"].values

        targets = None
        if not is_test:
            # 5 targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            targets = np.zeros((n_samples, 5, seq_len), dtype=np.float32)

        for idx, row in df.iterrows():
            # 1. Feature Extraction
            seq = row["sequence"]
            struct = row["structure"]
            loop = row["predicted_loop_type"]

            # One-hot encoding
            oh_seq = self.one_hot(seq, self.seq_map, 4)
            oh_struct = self.one_hot(struct, self.struct_map, 3)
            oh_loop = self.one_hot(loop, self.loop_map, 7)

            # Concatenate channels: (14, L)
            inputs[idx] = np.concatenate([oh_seq, oh_struct, oh_loop], axis=0)

            # 2. Partner Map
            partner_indices[idx] = self.get_partner_indices(struct)

            # 3. Targets (Training/Validation only)
            if not is_test:
                for t_i, col in enumerate(Config.TARGET_COLS):
                    val_str = row[col]
                    val_arr = parse_list_column(val_str)

                    # Pad to seq_len (107). Raw data is usually 68.
                    length = len(val_arr)
                    if length > 0:
                        targets[idx, t_i, :length] = val_arr

        return inputs, partner_indices, targets, ids

    def load_data(self, mode="train", load_cached_data=True):
        """
        Loads data from cache if available and requested, otherwise processes from CSV.
        """
        # Determine file paths based on mode
        if mode == "train":
            csv_path = Config.TRAIN_CSV
            cache_file = Config.TRAIN_CACHE_FILE
        elif mode == "val":
            csv_path = Config.VAL_CSV
            cache_file = Config.VAL_CACHE_FILE
        else:
            csv_path = Config.TEST_CSV
            cache_file = Config.TEST_CACHE_FILE

        cache_path = Config.get_cache_path(cache_file)

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {mode} data from cache: {cache_path}")
            try:
                data = np.load(cache_path, allow_pickle=True)
                inputs = data["inputs"]
                partner_indices = data["partner_indices"]
                ids = data["ids"]

                if mode == "test":
                    return inputs, partner_indices, None, ids
                else:
                    targets = data["targets"]
                    return inputs, partner_indices, targets, ids
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing from source...")

        # 2. Process from CSV
        print(f"Processing {mode} data from {csv_path}...")
        df = pd.read_csv(csv_path)
        is_test = mode == "test"
        inputs, partner_indices, targets, ids = self.process_df(df, is_test)

        # 3. Save Cache
        print(f"Saving {mode} data to cache: {cache_path}")
        save_dict = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}
        if not is_test:
            save_dict["targets"] = targets

        np.savez_compressed(cache_path, **save_dict)

        return inputs, partner_indices, targets, ids


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create Train, Val, and Test DataLoaders.
    """
    preloader = Preloader()

    # Train Loader
    train_inputs, train_partners, train_targets, train_ids = preloader.load_data(
        "train", load_cached_data
    )
    train_dataset = RNADataset(train_inputs, train_partners, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val Loader
    val_inputs, val_partners, val_targets, val_ids = preloader.load_data(
        "val", load_cached_data
    )
    val_dataset = RNADataset(val_inputs, val_partners, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Loader
    test_inputs, test_partners, _, test_ids = preloader.load_data(
        "test", load_cached_data
    )
    test_dataset = RNADataset(test_inputs, test_partners, None, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
