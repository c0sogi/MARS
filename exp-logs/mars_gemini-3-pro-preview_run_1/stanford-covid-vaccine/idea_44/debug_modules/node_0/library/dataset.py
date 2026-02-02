import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.features import (
    tokenize_sequence,
    tokenize_loop,
    get_signed_distance_vector,
    sinusoidal_encoding,
)


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, data, mode="train"):
        """
        Args:
            data (dict): Dictionary containing processed numpy arrays.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.seq_tokens = data["seq_tokens"]
        self.loop_tokens = data["loop_tokens"]
        self.dist_emb = data["dist_emb"]
        self.targets = data["targets"]
        self.ids = data["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to tensors
        # Inputs: LongTensor for tokens, FloatTensor for embeddings
        seq = torch.tensor(self.seq_tokens[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_tokens[idx], dtype=torch.long)
        dist = torch.tensor(self.dist_emb[idx], dtype=torch.float32)

        # Targets: FloatTensor
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "seq": seq,
            "loop": loop,
            "dist": dist,
            "targets": targets,
            "id": self.ids[idx],
        }


def process_dataframe(df, mode="train"):
    """
    Processes a Pandas DataFrame into numpy arrays for the dataset.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    pred_len = Config.PRED_LEN

    # Pre-allocate arrays
    seq_tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    loop_tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    dist_emb = np.zeros((num_samples, seq_len, Config.EMB_DIM_DIST), dtype=np.float32)

    # Targets: (N, 107, 3) - Padded with zeros
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    ids = df["id"].values.tolist()

    # Target columns to extract
    target_cols = Config.TARGET_COLS

    for i, row in df.iterrows():
        # 1. Sequence Tokenization
        seq_tokens[i] = tokenize_sequence(row["sequence"])

        # 2. Loop Type Tokenization
        loop_tokens[i] = tokenize_loop(row["predicted_loop_type"])

        # 3. Structure Distance Embedding
        # Calculate signed distances
        dists = get_signed_distance_vector(row["structure"])
        # Encode with sinusoidal embedding
        dist_emb[i] = sinusoidal_encoding(dists, Config.EMB_DIM_DIST)

        # 4. Targets
        if mode in ["train", "val"]:
            # Stack the 3 target lists: shape (3, 68) -> transpose to (68, 3)
            # Note: Parquet loads lists as numpy arrays or python lists
            t_list = []
            for col in target_cols:
                val = row[col]
                # Ensure it's a list/array of floats
                if isinstance(val, np.ndarray) or isinstance(val, list):
                    t_list.append(val)
                else:
                    # Fallback for unexpected format, though metadata ensures consistency
                    t_list.append(np.zeros(pred_len))

            # Stack: (3, 68)
            stacked_targets = np.vstack(t_list)
            # Transpose: (68, 3)
            stacked_targets = stacked_targets.T

            # Assign to the first 68 positions of the target array
            targets[i, :pred_len, :] = stacked_targets

            # Remaining positions (68 to 107) stay 0.0

    return {
        "seq_tokens": seq_tokens,
        "loop_tokens": loop_tokens,
        "dist_emb": dist_emb,
        "targets": targets,
        "ids": ids,
    }


def load_data(split="train", load_cached_data=True, debug=False):
    """
    Loads data for a specific split, using caching to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache.
        debug (bool): If True, loads only a small subset of data.

    Returns:
        RNADataset: The dataset object.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(Config.CACHE_DIR, f"cached_{split}.npz")

    # Determine source file
    if split == "train":
        source_path = Config.TRAIN_PATH
    elif split == "val":
        source_path = Config.VAL_PATH
    elif split == "test":
        source_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    data = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file) and not debug:
        try:
            # Allow pickle is required for object arrays (like string IDs)
            loaded = np.load(cache_file, allow_pickle=True)
            data = {
                "seq_tokens": loaded["seq_tokens"],
                "loop_tokens": loaded["loop_tokens"],
                "dist_emb": loaded["dist_emb"],
                "targets": loaded["targets"],
                "ids": loaded["ids"],
            }
            print(f"Loaded {split} data from cache: {cache_file}")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            data = None

    # 2. Process from scratch if needed
    if data is None:
        print(f"Processing {split} data from {source_path}...")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        df = pd.read_parquet(source_path)

        if debug:
            df = df.head(100)  # Small subset for debugging
            print("Debug mode: Processed only 100 samples.")

        data = process_dataframe(df, mode=split)

        # Save to cache (only if not debugging, to avoid overwriting full cache with debug data)
        if not debug:
            np.savez_compressed(
                cache_file,
                seq_tokens=data["seq_tokens"],
                loop_tokens=data["loop_tokens"],
                dist_emb=data["dist_emb"],
                targets=data["targets"],
                ids=data["ids"],
            )
            print(f"Saved {split} data to cache: {cache_file}")

    return RNADataset(data, mode=split)
