import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import compute_adjacency, compute_rwpe, compute_signed_distance

# Token Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Holds precomputed features and targets.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays for features and targets.
                              Keys: 'ids', 'sequence', 'loop_type', 'rwpe', 'distance', 'targets'
        """
        self.ids = data_dict["ids"]
        self.sequence = torch.from_numpy(data_dict["sequence"]).long()
        self.loop_type = torch.from_numpy(data_dict["loop_type"]).long()
        self.rwpe = torch.from_numpy(data_dict["rwpe"]).float()
        self.distance = torch.from_numpy(data_dict["distance"]).float()
        self.targets = torch.from_numpy(data_dict["targets"]).float()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "sequence": self.sequence[idx],
            "loop_type": self.loop_type[idx],
            "rwpe": self.rwpe[idx],
            "distance": self.distance[idx],
            "targets": self.targets[idx],
        }


def process_dataframe(df, config, is_test=False):
    """
    Processes a pandas DataFrame into numpy arrays suitable for the RNADataset.
    Computes RWPE and signed distances for each sample.
    """
    num_samples = len(df)
    seq_len = config.seq_len
    target_len = config.pred_len

    # Initialize arrays
    ids = df["id"].values
    seq_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    rwpe_arr = np.zeros(
        (num_samples, seq_len, len(config.rwpe_steps)), dtype=np.float32
    )
    dist_arr = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: (N, 68, 3) for train/val, dummy for test
    if not is_test:
        target_arr = np.zeros(
            (num_samples, target_len, len(config.target_cols)), dtype=np.float32
        )
    else:
        target_arr = np.zeros(
            (num_samples, target_len, len(config.target_cols)), dtype=np.float32
        )

    # Processing loop
    for i, row in df.iterrows():
        # 1. Sequence Encoding
        seq_chars = list(row["sequence"])
        seq_arr[i] = [
            SEQ_MAP.get(c, 0) for c in seq_chars
        ]  # Default to A if unknown, though unlikely

        # 2. Loop Type Encoding
        loop_chars = list(row["predicted_loop_type"])
        loop_arr[i] = [
            LOOP_MAP.get(c, 6) for c in loop_chars
        ]  # Default to X if unknown

        # 3. Structural Features
        structure = row["structure"]
        adj = compute_adjacency(structure, seq_len)

        # RWPE
        rwpe = compute_rwpe(adj, steps=config.rwpe_steps)
        rwpe_arr[i] = rwpe

        # Signed Distance
        dist = compute_signed_distance(structure, seq_len)
        dist_arr[i] = dist

        # 4. Targets
        if not is_test:
            # Stack the 3 target columns: reactivity, deg_Mg_pH10, deg_Mg_50C
            # Each is a list of length 68
            t_list = []
            for col in config.target_cols:
                val = row[col]
                # Ensure it's a list/array of length 68
                if len(val) != target_len:
                    # Fallback or error, but metadata guarantees consistency usually
                    val = list(val) + [0.0] * (target_len - len(val))
                t_list.append(val)

            # Transpose to (68, 3)
            target_arr[i] = np.array(t_list).T

    return {
        "ids": ids,
        "sequence": seq_arr,
        "loop_type": loop_arr,
        "rwpe": rwpe_arr,
        "distance": dist_arr,
        "targets": target_arr,
    }


def get_dataloaders(config, load_cached_data=True):
    """
    Orchestrates data loading, processing, caching, and DataLoader creation.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists for cache
    os.makedirs(config.working_dir, exist_ok=True)

    loaders = {}
    splits = [
        ("train", config.train_file, False),
        ("val", config.val_file, False),
        ("test", config.test_file, True),
    ]

    for split_name, file_path, is_test in splits:
        cache_path = os.path.join(config.working_dir, f"cached_{split_name}.npz")

        data_dict = None

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split_name} data from cache: {cache_path}")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {
                    "ids": loaded["ids"],
                    "sequence": loaded["sequence"],
                    "loop_type": loaded["loop_type"],
                    "rwpe": loaded["rwpe"],
                    "distance": loaded["distance"],
                    "targets": loaded["targets"],
                }
            except Exception as e:
                print(f"Failed to load cache for {split_name}: {e}")
                data_dict = None

        # 2. Process from Scratch if needed
        if data_dict is None:
            print(f"Processing {split_name} data from {file_path}...")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Metadata file not found: {file_path}")

            df = pd.read_parquet(file_path)

            # Debug mode: subsample
            if config.debug:
                df = df.iloc[:100].reset_index(drop=True)

            data_dict = process_dataframe(df, config, is_test=is_test)

            # Save to cache
            print(f"Saving {split_name} data to cache...")
            np.savez_compressed(
                cache_path,
                ids=data_dict["ids"],
                sequence=data_dict["sequence"],
                loop_type=data_dict["loop_type"],
                rwpe=data_dict["rwpe"],
                distance=data_dict["distance"],
                targets=data_dict["targets"],
            )

        # 3. Create Dataset and DataLoader
        dataset = RNADataset(data_dict)

        shuffle = split_name == "train"
        drop_last = split_name == "train"  # Drop last incomplete batch in training

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True if config.device == "cuda" else False,
            drop_last=drop_last,
        )
        loaders[split_name] = loader

    return loaders["train"], loaders["val"], loaders["test"]
