import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config, get_structure_distance_matrix


class RNADataset(Dataset):
    def __init__(self, df=None, mode="train", data_dict=None):
        self.mode = mode

        if data_dict is not None:
            # Load from pre-processed dictionary (Cache mode)
            self.ids = data_dict["ids"]
            self.seqs = data_dict["seqs"]
            self.loops = data_dict["loops"]
            self.dists = data_dict["dists"]
            if mode != "test" and "targets" in data_dict:
                self.targets = data_dict["targets"]
        else:
            # Process from DataFrame (Raw mode)
            self.ids = df["id"].values

            # Sequence Encoding: A:0, G:1, C:2, U:3
            self.seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
            self.seqs = []
            for s in df["sequence"].values:
                # Default to 0 (A) if unknown character is encountered
                self.seqs.append([self.seq_map.get(c, 0) for c in s])
            self.seqs = np.array(self.seqs, dtype=np.int64)

            # Loop Type Encoding: S, M, I, B, H, E, X
            self.loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
            self.loops = []
            for l in df["predicted_loop_type"].values:
                self.loops.append([self.loop_map.get(c, 0) for c in l])
            self.loops = np.array(self.loops, dtype=np.int64)

            # Signed Distance Encoding using helper from library.config
            self.dists = []
            for s in df["structure"].values:
                self.dists.append(get_structure_distance_matrix(s, Config.seq_len))
            self.dists = np.array(self.dists, dtype=np.float32)

            # Targets
            if self.mode != "test":
                # Stack targets: reactivity, deg_Mg_pH10, deg_Mg_50C
                # Filter strictly to these 3 targets as per the "Idea"
                def process_target(col_name):
                    raw = df[col_name].values
                    # Pad to seq_len (107) with 0.0
                    padded = np.zeros((len(raw), Config.seq_len), dtype=np.float32)
                    for i, arr in enumerate(raw):
                        length = len(arr)
                        safe_len = min(length, Config.seq_len)
                        padded[i, :safe_len] = arr[:safe_len]
                    return padded

                t1 = process_target("reactivity")
                t2 = process_target("deg_Mg_pH10")
                t3 = process_target("deg_Mg_50C")

                self.targets = np.stack([t1, t2, t3], axis=2)  # Shape: (N, 107, 3)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        out = {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "loop": torch.tensor(self.loops[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
        }
        if self.mode != "test" and hasattr(self, "targets"):
            out["y"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return out


def load_data(load_cached_data=True):
    """
    Loads data for train, val, and test splits.
    Implements caching using .npz files in Config.cache_dir.
    Strictly follows the logic: Try Load -> If Fail/False -> Process -> Save.
    """
    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    paths = [Config.train_data_path, Config.val_data_path, Config.test_data_path]
    datasets = {}

    for split, path in zip(splits, paths):
        cache_file = os.path.join(Config.cache_dir, f"cached_{split}.npz")

        loaded_from_cache = False

        # 1. Try to load from cache if requested
        if load_cached_data and os.path.exists(cache_file):
            try:
                print(f"Loading {split} data from cache: {cache_file}")
                data = np.load(cache_file, allow_pickle=True)
                data_dict = {
                    "ids": data["ids"],
                    "seqs": data["seqs"],
                    "loops": data["loops"],
                    "dists": data["dists"],
                }
                if split != "test" and "targets" in data:
                    data_dict["targets"] = data["targets"]

                datasets[split] = RNADataset(mode=split, data_dict=data_dict)
                loaded_from_cache = True
            except Exception as e:
                print(
                    f"Error loading cache for {split}: {e}. Re-processing from source."
                )
                loaded_from_cache = False

        # 2. Process from scratch if cache failed or was not requested
        if not loaded_from_cache:
            print(f"Processing {split} data from {path}...")
            df = pd.read_parquet(path)
            ds = RNADataset(df, mode=split)

            # Save to cache
            save_dict = {
                "ids": ds.ids,
                "seqs": ds.seqs,
                "loops": ds.loops,
                "dists": ds.dists,
            }
            if split != "test":
                save_dict["targets"] = ds.targets

            np.savez(cache_file, **save_dict)
            print(f"Saved {split} data to cache: {cache_file}")
            datasets[split] = ds

    return datasets["train"], datasets["val"], datasets["test"]
