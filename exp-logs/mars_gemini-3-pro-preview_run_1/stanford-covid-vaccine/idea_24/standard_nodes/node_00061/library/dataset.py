import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class RNADataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed data from cache if available.
        """
        self.mode = mode
        self.seq_len = Config.SEQ_LEN
        self.pred_len = Config.PRED_LEN

        # Determine input file path based on mode
        if mode == "train":
            self.data_path = Config.TRAIN_DATA_PATH
        elif mode == "val":
            self.data_path = Config.VAL_DATA_PATH
        elif mode == "test":
            self.data_path = Config.TEST_DATA_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load raw dataframe to get IDs and ensure alignment
        # We keep the dataframe in memory to access IDs easily,
        # but heavy features are stored in numpy arrays.
        self.df = pd.read_parquet(self.data_path)
        self.ids = self.df["id"].values

        # Initialize containers for features
        self.sequences = None
        self.loop_types = None
        self.pair_dists = None
        self.targets = None

        # Load or compute processed features
        self._prepare_data(load_cached_data)

    def _prepare_data(self, load_cached_data):
        """
        Loads processed features from cache or computes them from scratch.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"{self.mode}_data.npz")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {self.mode} data from {cache_path}")
            try:
                # We use allow_pickle=False to ensure we are loading strict numeric data
                data = np.load(cache_path, allow_pickle=False)
                self.sequences = data["sequences"]
                self.loop_types = data["loop_types"]
                self.pair_dists = data["pair_dists"]
                if self.mode in ["train", "val"]:
                    self.targets = data["targets"]
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing {self.mode} data from {self.data_path}...")

        # A. Sequence Tokenization
        token_map = Config.TOKEN_MAP
        raw_seqs = self.df["sequence"].values
        self.sequences = np.array(
            [[token_map[char] for char in seq] for seq in raw_seqs], dtype=np.int64
        )

        # B. Loop Type Tokenization
        loop_map = Config.LOOP_TYPE_MAP
        raw_loops = self.df["predicted_loop_type"].values
        self.loop_types = np.array(
            [[loop_map[char] for char in loop] for loop in raw_loops], dtype=np.int64
        )

        # C. Structure Parsing (Signed Pair Distances)
        structures = self.df["structure"].values
        num_samples = len(self.df)
        self.pair_dists = np.zeros((num_samples, self.seq_len), dtype=np.float32)

        for idx, struct in enumerate(structures):
            stack = []
            dists = np.zeros(self.seq_len, dtype=np.float32)
            for i, char in enumerate(struct):
                if char == "(":
                    stack.append(i)
                elif char == ")":
                    if stack:
                        j = stack.pop()
                        # j is opening (upstream, smaller index), i is closing (downstream, larger index)
                        # Distance relative to self:
                        # At j: paired with i -> dist = i - j (positive)
                        # At i: paired with j -> dist = j - i (negative)
                        dists[j] = float(i - j)
                        dists[i] = float(j - i)
            self.pair_dists[idx] = dists

        # D. Targets (Train/Val only)
        save_dict = {
            "sequences": self.sequences,
            "loop_types": self.loop_types,
            "pair_dists": self.pair_dists,
        }

        if self.mode in ["train", "val"]:
            # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
            # These columns contain lists of floats in the parquet file
            t_reactivity = np.vstack(self.df["reactivity"].values)
            t_deg_mg_ph10 = np.vstack(self.df["deg_Mg_pH10"].values)
            t_deg_mg_50c = np.vstack(self.df["deg_Mg_50C"].values)

            # Stack into (N, 68, 3)
            raw_targets = np.stack([t_reactivity, t_deg_mg_ph10, t_deg_mg_50c], axis=2)

            # Pad to SEQ_LEN (107) with zeros
            # We pad with 0.0. The loss function must mask these positions out.
            self.targets = np.zeros((num_samples, self.seq_len, 3), dtype=np.float32)
            self.targets[:, : self.pred_len, :] = raw_targets

            save_dict["targets"] = self.targets

        # 3. Save to cache
        np.savez(cache_path, **save_dict)
        print(f"Saved processed data to {cache_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'sequence': (L,),
                'loop_type': (L,),
                'pair_dist': (L,),
                'targets': (L, 3) [only if train/val],
                'id': str
            }
        """
        item = {
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.long),
            "loop_type": torch.tensor(self.loop_types[idx], dtype=torch.long),
            "pair_dist": torch.tensor(self.pair_dists[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item
