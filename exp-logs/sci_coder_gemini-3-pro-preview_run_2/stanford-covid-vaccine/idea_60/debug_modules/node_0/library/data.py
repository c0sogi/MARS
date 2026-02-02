import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class RNAProcessor:
    """
    Handles data processing for RNA sequences, including feature engineering
    and target parsing. Implements caching mechanism to store processed
    numpy arrays.
    """

    # Dictionaries for One-Hot Encoding
    SEQ_MAP = {c: i for i, c in enumerate("AGUC")}
    STRUCT_MAP = {c: i for i, c in enumerate("().")}
    LOOP_MAP = {c: i for i, c in enumerate("SMIBHEX")}

    def __init__(self):
        pass

    def get_pairs(self, structure):
        """
        Parses dot-bracket structure to find base pairs.
        Returns a mapping {index: partner_index}. Unpaired indices are not in the map.
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

    def encode_sequence(self, seq, map_dict, length):
        """One-hot encodes a sequence string."""
        arr = np.zeros((length, len(map_dict)), dtype=np.float32)
        for i, char in enumerate(seq):
            if i >= length:
                break
            if char in map_dict:
                arr[i, map_dict[char]] = 1.0
        return arr

    def generate_features(self, df):
        """
        Generates input features:
        1. Sequence One-Hot (4 channels)
        2. Structure One-Hot (3 channels)
        3. Loop Type One-Hot (7 channels)
        4. Partner Identity One-Hot (4 channels)

        Also generates Partner Index Map for the gathering operation.
        """
        num_samples = len(df)
        seq_len = Config.SEQ_LENGTH

        # Dimensions
        dim_seq = 4
        dim_struct = 3
        dim_loop = 7
        dim_partner = 4
        total_dim = dim_seq + dim_struct + dim_loop + dim_partner

        # Pre-allocate arrays
        inputs = np.zeros((num_samples, seq_len, total_dim), dtype=np.float32)
        partner_indices = np.full((num_samples, seq_len), -1, dtype=np.int32)

        for idx, row in df.iterrows():
            # Extract strings
            seq = row["sequence"]
            struct = row["structure"]
            loop = row["predicted_loop_type"]

            # 1. Basic One-Hot Encodings
            oh_seq = self.encode_sequence(seq, self.SEQ_MAP, seq_len)
            oh_struct = self.encode_sequence(struct, self.STRUCT_MAP, seq_len)
            oh_loop = self.encode_sequence(loop, self.LOOP_MAP, seq_len)

            # 2. Partner Analysis
            pairs = self.get_pairs(struct)

            # 3. Partner Identity & Index Map
            oh_partner = np.zeros((seq_len, dim_partner), dtype=np.float32)
            p_indices = np.full(seq_len, -1, dtype=np.int32)

            for i in range(seq_len):
                if i in pairs:
                    j = pairs[i]
                    p_indices[i] = j
                    # Get partner base identity
                    if j < len(seq):
                        partner_char = seq[j]
                        if partner_char in self.SEQ_MAP:
                            oh_partner[i, self.SEQ_MAP[partner_char]] = 1.0

            # Concatenate features: [Seq, Struct, Loop, Partner]
            sample_input = np.concatenate(
                [oh_seq, oh_struct, oh_loop, oh_partner], axis=1
            )

            inputs[idx] = sample_input
            partner_indices[idx] = p_indices

        return inputs, partner_indices

    def parse_targets(self, df, mode="train"):
        """
        Parses target columns.
        For training, pads the 68-length targets to 107 with 0.0 (Boundary Anchoring).
        """
        if mode == "test":
            return None

        num_samples = len(df)
        seq_len = Config.SEQ_LENGTH
        num_targets = len(Config.ALL_TARGETS)

        targets = np.zeros((num_samples, seq_len, num_targets), dtype=np.float32)

        for idx, row in df.iterrows():
            for t_i, col_name in enumerate(Config.ALL_TARGETS):
                # Parse stringified list
                try:
                    val_str = row[col_name]
                    if isinstance(val_str, str):
                        val_list = ast.literal_eval(val_str)
                    else:
                        val_list = (
                            val_str  # Already a list if processed upstream differently
                        )
                    val_arr = np.array(val_list, dtype=np.float32)
                except Exception:
                    # Fallback for empty or malformed
                    val_arr = np.zeros(Config.SEQ_SCORED, dtype=np.float32)

                # Assign to the first 68 positions (or whatever length is provided)
                len_valid = min(len(val_arr), seq_len)
                targets[idx, :len_valid, t_i] = val_arr[:len_valid]

                # The rest (68-107) remains 0.0, implementing Boundary Anchoring

        return targets

    def process(self, metadata_path, cache_path, mode="train", load_cached_data=True):
        """
        Main processing function with caching logic.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                inputs = data["inputs"]
                partner_indices = data["partner_indices"]
                ids = data["ids"]
                if mode != "test":
                    targets = data["targets"]
                    return inputs, partner_indices, targets, ids
                else:
                    return inputs, partner_indices, None, ids
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

        # 2. Compute from scratch
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Reset index to ensure alignment
        df = df.reset_index(drop=True)

        inputs, partner_indices = self.generate_features(df)
        ids = df["id"].values

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if mode != "test":
            targets = self.parse_targets(df, mode)
            np.savez_compressed(
                cache_path,
                inputs=inputs,
                partner_indices=partner_indices,
                targets=targets,
                ids=ids,
            )
            return inputs, partner_indices, targets, ids
        else:
            np.savez_compressed(
                cache_path, inputs=inputs, partner_indices=partner_indices, ids=ids
            )
            return inputs, partner_indices, None, ids


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA data.
    """

    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.partner_indices = torch.tensor(partner_indices, dtype=torch.long)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (Seq_Len, Channels)
        # partner_indices: (Seq_Len,)

        sample = {
            "inputs": self.inputs[idx],
            "partner_indices": self.partner_indices[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def get_loaders(load_cached_data=True, batch_size=None):
    """
    Generates DataLoaders for train, validation, and test sets.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    processor = RNAProcessor()

    # --- Train ---
    train_inputs, train_partners, train_targets, train_ids = processor.process(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        mode="train",
        load_cached_data=load_cached_data,
    )
    train_dataset = RNADataset(train_inputs, train_partners, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Validation ---
    val_inputs, val_partners, val_targets, val_ids = processor.process(
        Config.VAL_METADATA_PATH,
        Config.VAL_CACHE_PATH,
        mode="val",
        load_cached_data=load_cached_data,
    )
    val_dataset = RNADataset(val_inputs, val_partners, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    test_inputs, test_partners, _, test_ids = processor.process(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        mode="test",
        load_cached_data=load_cached_data,
    )
    test_dataset = RNADataset(test_inputs, test_partners, None, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
