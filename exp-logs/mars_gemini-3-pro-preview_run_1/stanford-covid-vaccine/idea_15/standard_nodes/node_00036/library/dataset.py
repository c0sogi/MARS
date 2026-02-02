import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config, get_structure_distance_matrix


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.

    Expects pre-processed numpy arrays for sequences, loops, distances, and targets.
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.ids = data_dict["ids"]
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.dist = data_dict["dist"]

        # Targets are only present for train/val sets
        if "targets" in data_dict and data_dict["targets"] is not None:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        seq_t = torch.tensor(self.seq[idx], dtype=torch.long)
        loop_t = torch.tensor(self.loop[idx], dtype=torch.long)
        dist_t = torch.tensor(self.dist[idx], dtype=torch.float)

        out = {"seq": seq_t, "loop": loop_t, "dist": dist_t, "id": self.ids[idx]}

        if self.targets is not None:
            out["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return out


def process_dataframe(df, mode="train"):
    """
    Processes a pandas DataFrame into a dictionary of numpy arrays.

    Args:
        df (pd.DataFrame): Input dataframe containing sequence, structure, etc.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing numpy arrays for 'seq', 'loop', 'dist', 'ids', and optionally 'targets'.
    """
    # Mappings
    token_map = {c: i for i, c in enumerate(["A", "G", "C", "U"])}
    loop_map = {c: i for i, c in enumerate(["S", "M", "I", "B", "H", "E", "X"])}

    # Pre-allocate arrays
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    seq_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    dist_arr = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Extract sequences and structures
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    for i in range(num_samples):
        # 1. Sequence Tokenization
        seq_arr[i] = [token_map.get(c, 0) for c in sequences[i]]

        # 2. Loop Tokenization
        loop_arr[i] = [loop_map.get(c, 0) for c in loops[i]]

        # 3. Structure Distance Calculation
        # Using the imported utility from config
        dist_arr[i] = get_structure_distance_matrix(structures[i], seq_len)

    data_dict = {"ids": ids, "seq": seq_arr, "loop": loop_arr, "dist": dist_arr}

    # 4. Targets (only for train/val)
    if mode in ["train", "val"]:
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        target_list = []
        for col in Config.TARGET_COLS:
            # Each column in df is a list/array of floats. Stack them.
            # We assume the metadata parquet has preserved them as lists/arrays.
            vals = np.vstack(df[col].values)
            target_list.append(vals)

        # Stack along the last dimension -> (N, 68, 3)
        targets_arr = np.stack(target_list, axis=2).astype(np.float32)
        data_dict["targets"] = targets_arr
    else:
        data_dict["targets"] = None

    return data_dict


def prepare_data(config=None, load_cached_data=True):
    """
    Loads data, processing from scratch if cache is missing or load_cached_data is False.
    Uses np.savez_compressed to avoid pickle.

    Args:
        config: Configuration object (optional, defaults to Config class).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {'train': RNADataset, 'val': RNADataset, 'test': RNADataset}
    """
    if config is None:
        config = Config()

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "train": os.path.join(config.WORKING_DIR, "train_data.npz"),
        "val": os.path.join(config.WORKING_DIR, "val_data.npz"),
        "test": os.path.join(config.WORKING_DIR, "test_data.npz"),
    }

    datasets = {}
    modes = ["train", "val", "test"]
    source_files = {
        "train": config.TRAIN_PARQUET,
        "val": config.VAL_PARQUET,
        "test": config.TEST_PARQUET,
    }

    for mode in modes:
        cache_file = cache_paths[mode]
        data_dict = None

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                print(f"Loading {mode} data from cache: {cache_file}")
                loaded = np.load(
                    cache_file, allow_pickle=True
                )  # allow_pickle=True needed for object arrays (ids)

                # Reconstruct dictionary from NpzFile
                data_dict = {
                    "ids": loaded["ids"],
                    "seq": loaded["seq"],
                    "loop": loaded["loop"],
                    "dist": loaded["dist"],
                }
                if "targets" in loaded:
                    data_dict["targets"] = loaded["targets"]
                else:
                    data_dict["targets"] = None

            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                data_dict = None

        # 2. Process from Scratch if needed
        if data_dict is None:
            print(f"Processing {mode} data from scratch...")
            df = pd.read_parquet(source_files[mode])
            data_dict = process_dataframe(df, mode=mode)

            # Save to cache
            save_dict = {
                "ids": data_dict["ids"],
                "seq": data_dict["seq"],
                "loop": data_dict["loop"],
                "dist": data_dict["dist"],
            }
            if data_dict["targets"] is not None:
                save_dict["targets"] = data_dict["targets"]

            np.savez_compressed(cache_file, **save_dict)
            print(f"Saved processed {mode} data to {cache_file}")

        # 3. Create Dataset
        datasets[mode] = RNADataset(data_dict, mode=mode)

    return datasets
