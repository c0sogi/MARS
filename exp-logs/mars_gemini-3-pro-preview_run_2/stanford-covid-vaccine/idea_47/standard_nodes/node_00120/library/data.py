import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataProcessor:
    """
    Handles loading, processing, and caching of RNA data for the PF-DRN model.
    Generates One-Hot encodings, Partner Indices, and Partner Identity features.
    """

    def __init__(self):
        # Mappings for One-Hot Encoding
        self.seq_map = {c: i for i, c in enumerate("AGCU")}
        self.struct_map = {c: i for i, c in enumerate(".()")}
        self.loop_map = {c: i for i, c in enumerate("SMIBHEX")}

        # Reverse map for partner identity (Index -> OneHot) is implicit via array indexing

    def parse_structure_pairs(self, structure):
        """
        Parses dot-bracket structure to find base pairs.
        Returns an array of indices where arr[i] = j means i is paired with j.
        Unpaired bases are marked with -1.
        """
        L = len(structure)
        partners = np.full(L, -1, dtype=np.int32)
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

    def get_one_hot(self, sequence, mapping, length):
        """
        Converts a string sequence into a One-Hot encoded array (Length, Channels).
        """
        num_channels = len(mapping)
        encoding = np.zeros((length, num_channels), dtype=np.float32)

        for i, char in enumerate(sequence):
            if i >= length:
                break
            if char in mapping:
                encoding[i, mapping[char]] = 1.0

        return encoding

    def process_data(self, df, mode="train"):
        """
        Iterates over the dataframe and generates features and targets.
        """
        num_samples = len(df)
        seq_len = Config.SEQ_LENGTH

        # Feature containers
        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
        inputs = np.zeros((num_samples, seq_len, 18), dtype=np.float32)
        partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

        # Target container (Num_Samples, Seq_Len, 5)
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

        # IDs for submission
        ids = []

        for idx, row in df.iterrows():
            sequence = row["sequence"]
            structure = row["structure"]
            loop_type = row["predicted_loop_type"]

            # 1. Base Features
            oh_seq = self.get_one_hot(sequence, self.seq_map, seq_len)  # (L, 4)
            oh_struct = self.get_one_hot(structure, self.struct_map, seq_len)  # (L, 3)
            oh_loop = self.get_one_hot(loop_type, self.loop_map, seq_len)  # (L, 7)

            # 2. Partner Indices
            p_indices = self.parse_structure_pairs(structure)
            partner_indices[idx] = p_indices

            # 3. Partner Identity Feature
            # Create (L, 4) where row i is the one-hot sequence of partner j
            oh_partner = np.zeros((seq_len, 4), dtype=np.float32)

            # Vectorized assignment for efficiency
            # Mask for paired bases
            paired_mask = p_indices != -1
            # Get indices of partners
            valid_partners = p_indices[paired_mask]
            # Assign features: row i gets row j of oh_seq
            oh_partner[paired_mask] = oh_seq[valid_partners]

            # Concatenate all features
            # Shape: (L, 4+3+7+4) = (L, 18)
            sample_input = np.concatenate(
                [oh_seq, oh_struct, oh_loop, oh_partner], axis=1
            )
            inputs[idx] = sample_input

            # 4. Targets
            if mode in ["train", "val"]:
                for t_i, col in enumerate(Config.TARGET_COLS):
                    val_str = row[col]
                    try:
                        # Parse string list
                        val_list = ast.literal_eval(val_str)
                        # Assign to first len(val_list) positions (usually 68)
                        length = min(len(val_list), seq_len)
                        targets[idx, :length, t_i] = val_list[:length]
                    except:
                        pass  # Keep zeros if parsing fails or empty

            ids.append(row["id"])

        return {
            "inputs": inputs,
            "partner_indices": partner_indices,
            "targets": targets,
            "ids": np.array(ids),
        }

    def load_data(self, mode="train", load_cached_data=True):
        """
        Loads data from cache or processes from CSV.

        Args:
            mode: 'train', 'val', or 'test'
            load_cached_data: Whether to try loading from .npz cache

        Returns:
            Dictionary containing inputs, partner_indices, targets, ids
        """
        cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data_pf_drn_v1.npz")

        # 1. Try Cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {mode} data from cache: {cache_file}")
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                return {
                    "inputs": loaded["inputs"],
                    "partner_indices": loaded["partner_indices"],
                    "targets": loaded["targets"],
                    "ids": loaded["ids"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from CSV
        print(f"Processing {mode} data from metadata...")
        if mode == "train":
            csv_path = Config.TRAIN_METADATA
        elif mode == "val":
            csv_path = Config.VAL_METADATA
        else:
            csv_path = Config.TEST_METADATA

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        data = self.process_data(df, mode=mode)

        # 3. Save Cache
        print(f"Saving {mode} data to cache: {cache_file}")
        np.savez_compressed(
            cache_file,
            inputs=data["inputs"],
            partner_indices=data["partner_indices"],
            targets=data["targets"],
            ids=data["ids"],
        )

        return data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA data.
    """

    def __init__(self, data_dict):
        self.inputs = torch.from_numpy(data_dict["inputs"]).float()  # (N, L, 18)
        self.partner_indices = torch.from_numpy(
            data_dict["partner_indices"]
        ).long()  # (N, L)
        self.targets = torch.from_numpy(data_dict["targets"]).float()  # (N, L, 5)
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Retrieve tensors
        x = self.inputs[idx]  # (L, 18)
        p_idx = self.partner_indices[idx]  # (L,)
        y = self.targets[idx]  # (L, 5)

        # Permute input to (C, L) for Conv1d
        x = x.permute(1, 0)  # (18, L)

        # Handle Partner Indices for Gathering
        # Unpaired bases have index -1. We replace -1 with 0 to avoid index errors during gather.
        # We create a mask where 1 indicates a valid pair, 0 indicates unpaired.
        pairing_mask = (p_idx != -1).float()  # (L,)
        safe_p_idx = p_idx.clone()
        safe_p_idx[safe_p_idx == -1] = 0

        return {
            "inputs": x,
            "partner_indices": safe_p_idx,
            "pairing_mask": pairing_mask,
            "targets": y,
            "id": self.ids[idx],
        }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    processor = RNADataProcessor()

    # Load Data
    train_data = processor.load_data("train", load_cached_data)
    val_data = processor.load_data("val", load_cached_data)
    test_data = processor.load_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
