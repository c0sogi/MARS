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
    Handles feature generation and data preprocessing for RNA sequences.
    """

    def __init__(self):
        self.seq_map = Config.TOKEN_TO_INT_SEQ
        self.struct_map = Config.TOKEN_TO_INT_STRUCT
        self.loop_map = Config.TOKEN_TO_INT_LOOP

    def get_partner_map(self, structure):
        """
        Generates a mapping where map[i] = j if base i is paired with j, else -1.
        """
        length = len(structure)
        partner_map = np.full(length, -1, dtype=np.int32)
        stack = []

        for i, char in enumerate(structure):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    j = stack.pop()
                    partner_map[i] = j
                    partner_map[j] = i
        return partner_map

    def one_hot_encode(self, sequence, mapping, vocab_size):
        """
        One-hot encodes a sequence string based on a mapping.
        Returns shape (Length, Vocab_Size).
        """
        seq_len = len(sequence)
        encoding = np.zeros((seq_len, vocab_size), dtype=np.float32)
        for i, char in enumerate(sequence):
            if char in mapping:
                encoding[i, mapping[char]] = 1.0
        return encoding

    def get_partner_identity(self, sequence, partner_map):
        """
        Generates one-hot encoding of the partner base.
        If unpaired, returns zero vector.
        Returns shape (Length, 4).
        """
        seq_len = len(sequence)
        encoding = np.zeros((seq_len, 4), dtype=np.float32)
        char_to_int = self.seq_map

        for i in range(seq_len):
            partner_idx = partner_map[i]
            if partner_idx != -1:
                partner_char = sequence[partner_idx]
                if partner_char in char_to_int:
                    encoding[i, char_to_int[partner_char]] = 1.0
        return encoding

    def process_data(self, df, is_test=False):
        """
        Processes a dataframe into numpy arrays for inputs and targets.
        """
        num_samples = len(df)
        seq_len = Config.SEQ_LEN

        # Initialize containers
        X_seq = np.zeros(
            (num_samples, seq_len, Config.VOCAB_SIZE_SEQ), dtype=np.float32
        )
        X_struct = np.zeros(
            (num_samples, seq_len, Config.VOCAB_SIZE_STRUCT), dtype=np.float32
        )
        X_loop = np.zeros(
            (num_samples, seq_len, Config.VOCAB_SIZE_LOOP), dtype=np.float32
        )
        X_partner = np.zeros(
            (num_samples, seq_len, Config.VOCAB_SIZE_PARTNER), dtype=np.float32
        )

        partner_maps = np.zeros((num_samples, seq_len), dtype=np.int32)
        Y = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)
        ids = []

        # Helper to safely parse stringified lists
        def parse_col(val):
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except:
                    return []
            return val if isinstance(val, list) else []

        for idx, row in df.iterrows():
            # 1. Inputs
            seq = row["sequence"]
            struct = row["structure"]
            loop = row["predicted_loop_type"]

            # One-Hot Encoding
            X_seq[idx] = self.one_hot_encode(seq, self.seq_map, Config.VOCAB_SIZE_SEQ)
            X_struct[idx] = self.one_hot_encode(
                struct, self.struct_map, Config.VOCAB_SIZE_STRUCT
            )
            X_loop[idx] = self.one_hot_encode(
                loop, self.loop_map, Config.VOCAB_SIZE_LOOP
            )

            # Partner Logic
            p_map = self.get_partner_map(struct)
            partner_maps[idx] = p_map
            X_partner[idx] = self.get_partner_identity(seq, p_map)

            ids.append(row["id"])

            # 2. Targets and Masks
            scored_len = row["seq_scored"]

            if not is_test:
                # Parse targets
                for t_i, col_name in enumerate(Config.TARGET_COLS):
                    val_list = parse_col(row[col_name])
                    length = min(len(val_list), seq_len)
                    Y[idx, :length, t_i] = val_list[:length]

                # Mask valid positions
                masks[idx, :scored_len] = 1.0
            else:
                # For test, we just set the mask for potential evaluation/submission logic
                masks[idx, :scored_len] = 1.0

        return {
            "X_seq": X_seq,
            "X_struct": X_struct,
            "X_loop": X_loop,
            "X_partner": X_partner,
            "partner_maps": partner_maps,
            "Y": Y,
            "masks": masks,
            "ids": np.array(ids),
        }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.X_seq = torch.from_numpy(data_dict["X_seq"])
        self.X_struct = torch.from_numpy(data_dict["X_struct"])
        self.X_loop = torch.from_numpy(data_dict["X_loop"])
        self.X_partner = torch.from_numpy(data_dict["X_partner"])
        self.partner_maps = torch.from_numpy(data_dict["partner_maps"]).long()
        self.Y = torch.from_numpy(data_dict["Y"])
        self.masks = torch.from_numpy(data_dict["masks"])
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Concatenate inputs: (L, C_seq) + (L, C_struct) + ... -> (L, Total_C)
        inputs = torch.cat(
            [
                self.X_seq[idx],
                self.X_struct[idx],
                self.X_loop[idx],
                self.X_partner[idx],
            ],
            dim=1,
        )

        # Transpose to (Total_C, L) for Conv1d input
        inputs = inputs.permute(1, 0)

        return {
            "inputs": inputs,
            "partner_map": self.partner_maps[idx],
            "targets": self.Y[idx],
            "mask": self.masks[idx],
            "id": self.ids[idx],
        }


def get_data(mode="train", load_cached_data=True):
    """
    Loads data for train, val, or test. Handles caching using .npz files.
    """
    processor = RNAProcessor()

    if mode == "train":
        csv_path = Config.TRAIN_CSV
        cache_path = Config.TRAIN_CACHE
        is_test = False
    elif mode == "val":
        csv_path = Config.VAL_CSV
        cache_path = Config.VAL_CACHE
        is_test = False
    elif mode == "test":
        csv_path = Config.TEST_CSV
        cache_path = Config.TEST_CACHE
        is_test = True
    else:
        raise ValueError("Mode must be 'train', 'val', or 'test'")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data_dict = {
                "X_seq": loaded["X_seq"],
                "X_struct": loaded["X_struct"],
                "X_loop": loaded["X_loop"],
                "X_partner": loaded["X_partner"],
                "partner_maps": loaded["partner_maps"],
                "Y": loaded["Y"],
                "masks": loaded["masks"],
                "ids": loaded["ids"],
            }
            return data_dict
        except Exception as e:
            pass  # Fallback to processing if cache load fails

    # Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file {csv_path} not found.")

    df = pd.read_csv(csv_path)
    data_dict = processor.process_data(df, is_test=is_test)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_seq=data_dict["X_seq"],
        X_struct=data_dict["X_struct"],
        X_loop=data_dict["X_loop"],
        X_partner=data_dict["X_partner"],
        partner_maps=data_dict["partner_maps"],
        Y=data_dict["Y"],
        masks=data_dict["masks"],
        ids=data_dict["ids"],
    )

    return data_dict


def get_dataloaders(load_cached_data=True, batch_size=None):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    train_data = get_data("train", load_cached_data)
    val_data = get_data("val", load_cached_data)
    test_data = get_data("test", load_cached_data)

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
