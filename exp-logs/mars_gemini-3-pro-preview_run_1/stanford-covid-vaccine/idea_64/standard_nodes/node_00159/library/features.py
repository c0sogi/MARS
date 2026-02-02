import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Maps characters to integer indices based on a provided vocabulary map.
    """

    def __init__(self, vocab_map):
        self.vocab_map = vocab_map

    def transform(self, sequences):
        """
        Converts a list of strings into a numpy array of integer indices.

        Args:
            sequences (list or pd.Series): List of strings (e.g., RNA sequences).

        Returns:
            np.ndarray: Array of shape (N, L) with integer indices.
        """
        # Assume all sequences have the same length (107)
        if len(sequences) == 0:
            return np.array([])

        seq_len = len(sequences[0])
        n_samples = len(sequences)

        # Initialize output array
        indices = np.zeros((n_samples, seq_len), dtype=np.int32)

        # Vectorized processing is tricky with strings in numpy,
        # but simple iteration is fast enough for ~2k samples * 107 chars.
        # For significantly larger datasets, one might use np.frombuffer or similar tricks.
        for i, seq in enumerate(sequences):
            indices[i] = [self.vocab_map.get(char, 0) for char in seq]

        return indices


def compute_pair_distance(structures):
    """
    Parses dot-bracket structure strings to compute signed pair distances.

    Distance = paired_index - current_index
    If unpaired, distance is 0.

    Args:
        structures (list or pd.Series): List of structure strings (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (N, L) containing signed integer distances.
    """
    if len(structures) == 0:
        return np.array([])

    n_samples = len(structures)
    seq_len = len(structures[0])

    dists = np.zeros((n_samples, seq_len), dtype=np.int32)

    for idx, struct_str in enumerate(structures):
        stack = []
        for i, char in enumerate(struct_str):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    j = stack.pop()
                    # j is opening index, i is closing index
                    # Distance at j (opening): i - j (positive)
                    # Distance at i (closing): j - i (negative)
                    dists[idx, j] = i - j
                    dists[idx, i] = j - i

    return dists


def get_sinusoidal_encoding(dists, embed_dim):
    """
    Generates fixed sinusoidal embeddings for signed distances.

    PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Args:
        dists (torch.Tensor or np.ndarray): Input distances of shape (..., SeqLen).
        embed_dim (int): Dimension of the embedding.

    Returns:
        torch.Tensor: Embeddings of shape (..., SeqLen, embed_dim).
    """
    if not isinstance(dists, torch.Tensor):
        dists = torch.tensor(dists, dtype=torch.float32)
    else:
        dists = dists.to(dtype=torch.float32)

    # Flatten to simplify broadcasting
    original_shape = dists.shape
    dists_flat = dists.view(-1, 1)  # (Total_Elements, 1)

    # Compute division term: 10000^(2i/dim)
    # We compute this for i = 0, 1, ..., dim/2 - 1
    half_dim = embed_dim // 2
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32, device=dists.device)
        * (-np.log(10000.0) / embed_dim)
    )  # Shape: (half_dim,)

    # Compute arguments: pos * div_term
    # Shape: (Total_Elements, half_dim)
    args = dists_flat * div_term

    # Initialize embedding tensor
    pe = torch.zeros(dists_flat.size(0), embed_dim, device=dists.device)

    # Apply Sin and Cos
    pe[:, 0::2] = torch.sin(args)
    pe[:, 1::2] = torch.cos(args)

    # Reshape back to original dimensions + embedding dim
    return pe.view(*original_shape, embed_dim)


def load_data(split="train", load_cached_data=True):
    """
    Loads, processes, and caches data for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        dict: Dictionary containing numpy arrays:
            - 'seq': (N, 107) int32
            - 'loop': (N, 107) int32
            - 'dist': (N, 107) int32
            - 'targets': (N, 68, 3) float32 (only for train/val)
            - 'ids': list of strings
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        # print(f"Loading {split} data from cache: {cache_file}")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            data = {
                "seq": loaded["seq"],
                "loop": loaded["loop"],
                "dist": loaded["dist"],
                "ids": loaded["ids"].tolist(),
            }
            if "targets" in loaded:
                data["targets"] = loaded["targets"]
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Process from scratch
    # Determine source file
    if split == "train":
        source_path = Config.TRAIN_FILE
    elif split == "val":
        source_path = Config.VAL_FILE
    elif split == "test":
        source_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load Parquet
    df = pd.read_parquet(source_path)

    # Initialize Tokenizers
    # Maps based on Config
    seq_vocab = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_vocab = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}

    seq_tokenizer = Tokenizer(seq_vocab)
    loop_tokenizer = Tokenizer(loop_vocab)

    # Process Inputs
    seq_data = seq_tokenizer.transform(df["sequence"].tolist())
    loop_data = loop_tokenizer.transform(df["predicted_loop_type"].tolist())
    dist_data = compute_pair_distance(df["structure"].tolist())
    ids = df["id"].values

    result = {"seq": seq_data, "loop": loop_data, "dist": dist_data, "ids": ids}

    # Process Targets (only for train/val)
    if split in ["train", "val"]:
        # Extract target columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # These are stored as lists/arrays in the dataframe cells
        # We assume they are of length 68

        t1 = np.vstack(df["reactivity"].values)
        t2 = np.vstack(df["deg_Mg_pH10"].values)
        t3 = np.vstack(df["deg_Mg_50C"].values)

        # Stack along the last dimension to get (N, 68, 3)
        targets = np.stack([t1, t2, t3], axis=-1).astype(np.float32)
        result["targets"] = targets

    # 3. Save to cache
    save_dict = {
        "seq": result["seq"],
        "loop": result["loop"],
        "dist": result["dist"],
        "ids": result["ids"],
    }
    if "targets" in result:
        save_dict["targets"] = result["targets"]

    np.savez_compressed(cache_file, **save_dict)
    # print(f"Saved {split} data to cache: {cache_file}")

    return result
