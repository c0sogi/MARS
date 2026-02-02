import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mappings for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Handles loading, feature generation, and caching of RNA data.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.
        """
        self.mode = mode
        self.seq_len = Config.SEQ_LENGTH

        # Determine paths based on mode
        if mode == "train":
            self.csv_path = Config.TRAIN_CSV
            self.cache_path = Config.TRAIN_CACHE
        elif mode == "val":
            self.csv_path = Config.VAL_CSV
            self.cache_path = Config.VAL_CACHE
        elif mode == "test":
            self.csv_path = Config.TEST_CSV
            self.cache_path = Config.TEST_CACHE
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Load or Process Data
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached {mode} data from {self.cache_path}...")
            data = np.load(self.cache_path)
            self.features = data["features"]
            self.partner_indices = data["partner_indices"]
            self.targets = data["targets"]
            self.ids = data["ids"]
        else:
            print(f"Processing {mode} data from {self.csv_path}...")
            self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads CSV, generates features and targets, and saves to .npz cache.
        """
        df = pd.read_csv(self.csv_path)

        # Initialize containers
        num_samples = len(df)
        features = np.zeros(
            (num_samples, self.seq_len, Config.INPUT_DIM), dtype=np.float32
        )
        partner_indices = np.full((num_samples, self.seq_len), -1, dtype=np.int32)
        targets = np.zeros(
            (num_samples, self.seq_len, Config.OUTPUT_DIM), dtype=np.float32
        )
        ids = df["id"].values

        # Process each sample
        for i, row in df.iterrows():
            seq = row["sequence"]
            struct = row["structure"]
            loop = row["predicted_loop_type"]

            # 1. Generate Static Features & Partner Indices
            feats, p_indices = self._generate_features(seq, struct, loop)
            features[i] = feats
            partner_indices[i] = p_indices

            # 2. Process Targets (only for train/val)
            if self.mode != "test":
                for t_idx, col in enumerate(Config.TARGET_COLS):
                    # Parse stringified list: "[0.1, 0.2, ...]"
                    val_list = ast.literal_eval(row[col])
                    # Fill the first len(val_list) positions (usually 68)
                    length = len(val_list)
                    targets[i, :length, t_idx] = val_list
            # For test, targets remain zeros

        # Save to cache
        np.savez_compressed(
            self.cache_path,
            features=features,
            partner_indices=partner_indices,
            targets=targets,
            ids=ids,
        )

        # Assign to self
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids
        print(f"Saved processed data to {self.cache_path}")

    def _generate_features(self, sequence, structure, loop_type):
        """
        Generates the input feature tensor and partner indices for a single sample.

        Features (18 dim):
        - 0-3: Sequence One-Hot (A, G, C, U)
        - 4-6: Structure One-Hot ((, ), .)
        - 7-13: Loop Type One-Hot (S, M, I, B, H, E, X)
        - 14-17: Partner Identity One-Hot (A, G, C, U of the paired base)
        """
        length = len(sequence)

        # 1. Basic One-Hot Encodings
        seq_oh = np.zeros((length, 4), dtype=np.float32)
        struct_oh = np.zeros((length, 3), dtype=np.float32)
        loop_oh = np.zeros((length, 7), dtype=np.float32)

        for j, char in enumerate(sequence):
            if char in SEQ_MAP:
                seq_oh[j, SEQ_MAP[char]] = 1.0

        for j, char in enumerate(structure):
            if char in STRUCT_MAP:
                struct_oh[j, STRUCT_MAP[char]] = 1.0

        for j, char in enumerate(loop_type):
            if char in LOOP_MAP:
                loop_oh[j, LOOP_MAP[char]] = 1.0

        # 2. Partner Indices Calculation
        p_indices = np.full(length, -1, dtype=np.int32)
        stack = []
        for j, char in enumerate(structure):
            if char == "(":
                stack.append(j)
            elif char == ")":
                if stack:
                    partner = stack.pop()
                    p_indices[j] = partner
                    p_indices[partner] = j

        # 3. Explicit Partner Identity
        # If paired, take the one-hot of the partner. If unpaired, zeros.
        partner_identity = np.zeros((length, 4), dtype=np.float32)
        for j in range(length):
            pidx = p_indices[j]
            if pidx != -1:
                partner_identity[j] = seq_oh[pidx]

        # 4. Concatenate
        # Dimensions: 4 + 3 + 7 + 4 = 18
        combined_features = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_identity], axis=1
        )

        return combined_features, p_indices

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            features: (Seq_Len, 18)
            partner_indices: (Seq_Len,)
            targets: (Seq_Len, 5)
        """
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.partner_indices[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = RNADataset(mode="train", load_cached_data=load_cached_data)
    val_dataset = RNADataset(mode="val", load_cached_data=load_cached_data)
    test_dataset = RNADataset(mode="test", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
