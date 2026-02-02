import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_parsed_metadata


def get_couples(structure):
    """
    Generates a mapping of paired bases from the dot-bracket structure.
    Returns a numpy array where index i contains the index of the partner of base i,
    or -1 if unpaired.
    """
    partner_map = np.full(len(structure), -1, dtype=int)
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


def get_partner_identity(sequence, partner_map):
    """
    Generates one-hot encoded features for the partner base identity.
    Shape: (Length, 5) -> A, G, U, C, NoPartner
    """
    seq_vocab = {"A": 0, "G": 1, "U": 2, "C": 3}
    length = len(sequence)
    # 5 channels: A, G, U, C, None
    partner_feat = np.zeros((length, 5), dtype=np.float32)

    for i in range(length):
        partner_idx = partner_map[i]
        if partner_idx != -1:
            partner_base = sequence[partner_idx]
            if partner_base in seq_vocab:
                partner_feat[i, seq_vocab[partner_base]] = 1.0
        else:
            # No partner
            partner_feat[i, 4] = 1.0

    return partner_feat


def one_hot(seq, vocab):
    res = np.zeros((len(seq), len(vocab)), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in vocab:
            res[i, vocab[char]] = 1.0
    return res


def process_data(df, mode="train"):
    """
    Processes the dataframe into numpy arrays for inputs, partner maps, and targets.
    """
    seq_vocab = {"A": 0, "G": 1, "U": 2, "C": 3}
    struct_vocab = {"(": 0, ")": 1, ".": 2}
    loop_vocab = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = []
    inputs = []
    partner_indices_list = []
    targets = []

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Basic One-Hot Features
        oh_seq = one_hot(seq, seq_vocab)  # (L, 4)
        oh_struct = one_hot(struct, struct_vocab)  # (L, 3)
        oh_loop = one_hot(loop, loop_vocab)  # (L, 7)

        # 2. Partner Map
        pmap = get_couples(struct)
        partner_indices_list.append(pmap)

        # 3. Partner Identity Features
        partner_id_feat = get_partner_identity(seq, pmap)  # (L, 5)

        # 4. Concatenate Static Inputs
        # Total channels: 4 + 3 + 7 + 5 = 19
        sample_input = np.concatenate(
            [oh_seq, oh_struct, oh_loop, partner_id_feat], axis=1
        )
        inputs.append(sample_input)
        ids.append(row["id"])

        # 5. Targets (if not test)
        if mode != "test":
            t_list = []
            for col in Config.TARGET_COLS:
                # Data is already parsed by get_parsed_metadata, so it should be np.array or list
                val = row[col]
                # Ensure it's an array
                if isinstance(val, (list, tuple)):
                    arr = np.array(val, dtype=np.float32)
                else:
                    arr = val

                # Pad to SEQ_LENGTH (107)
                padded = np.zeros(Config.SEQ_LENGTH, dtype=np.float32)
                len_arr = len(arr)
                if len_arr > 0:
                    padded[:len_arr] = arr
                t_list.append(padded)

            sample_target = np.stack(t_list, axis=1)  # (L, 5)
            targets.append(sample_target)

    inputs = np.array(inputs, dtype=np.float32)
    partner_indices = np.array(partner_indices_list, dtype=np.int64)

    if mode != "test":
        targets = np.array(targets, dtype=np.float32)
        return ids, inputs, partner_indices, targets
    else:
        return ids, inputs, partner_indices


def get_dataset(mode="train", load_cached_data=True):
    """
    Retrieves the dataset, using caching to speed up subsequent loads.
    Cache key: {mode}_data_stabilized_recurrent_v1.npz
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_filename = f"{mode}_data_stabilized_recurrent_v1.npz"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            if mode == "test":
                return data["ids"], data["inputs"], data["partner_indices"]
            else:
                return (
                    data["ids"],
                    data["inputs"],
                    data["partner_indices"],
                    data["targets"],
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {mode} data from metadata...")
    # Use library utility to load and parse metadata
    df = get_parsed_metadata(mode=mode)

    if mode == "test":
        ids, inputs, p_indices = process_data(df, mode)
        # 3. Save to cache
        np.savez(cache_path, ids=ids, inputs=inputs, partner_indices=p_indices)
        return ids, inputs, p_indices
    else:
        ids, inputs, p_indices, targets = process_data(df, mode)
        # 3. Save to cache
        np.savez(
            cache_path,
            ids=ids,
            inputs=inputs,
            partner_indices=p_indices,
            targets=targets,
        )
        return ids, inputs, p_indices, targets


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        pmap = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, pmap, y

        return x, pmap
