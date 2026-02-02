import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Handles tokenization of sequences and structures, and formatting of targets.
    Implements caching to speed up data loading.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from cached .pt file.
        """
        self.split = split
        self.seq_len = Config.SEQ_LEN
        self.pred_len = Config.PRED_LEN

        # Determine file paths based on split
        if split == "train":
            self.raw_path = Config.TRAIN_PATH
            self.cache_path = Config.CACHE_TRAIN
        elif split == "val":
            self.raw_path = Config.VAL_PATH
            self.cache_path = Config.CACHE_VAL
        elif split == "test":
            self.raw_path = Config.TEST_PATH
            self.cache_path = Config.CACHE_TEST
        else:
            raise ValueError(
                f"Unknown split: {split}. Must be 'train', 'val', or 'test'."
            )

        # Load data (from cache or process raw)
        self.data = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads data from cache if available, otherwise processes raw parquet file.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                # print(f"Loading {self.split} data from cache: {self.cache_path}")
                return torch.load(self.cache_path)
            except Exception as e:
                print(f"Error loading cache for {self.split}: {e}. Reprocessing...")

        # 2. Process from raw parquet
        # print(f"Processing raw {self.split} data from: {self.raw_path}")
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

        df = pd.read_parquet(self.raw_path)

        # --- Tokenization ---
        map_seq = Config.get_token_map_seq()
        map_struct = Config.get_token_map_struct()
        map_loop = Config.get_token_map_loop()

        def tokenize(sequence, mapping):
            # Map chars to ints, default to 0 if unknown (though data should be clean)
            return [mapping.get(char, 0) for char in sequence]

        # Convert columns to list of lists
        seq_tokens = [tokenize(s, map_seq) for s in df["sequence"]]
        struct_tokens = [tokenize(s, map_struct) for s in df["structure"]]
        loop_tokens = [tokenize(s, map_loop) for s in df["predicted_loop_type"]]

        # Convert to LongTensors
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long)
        struct_tensor = torch.tensor(struct_tokens, dtype=torch.long)
        loop_tensor = torch.tensor(loop_tokens, dtype=torch.long)

        # --- Target & Mask Processing ---
        n_samples = len(df)
        ids = df["id"].tolist()

        if self.split == "test":
            # Test set has no targets. Create dummy tensors for consistency.
            # Shape: (N, 107, 5)
            target_tensor = torch.zeros(
                (n_samples, self.seq_len, Config.NUM_TARGETS), dtype=torch.float32
            )
            # Mask: All zeros (or irrelevant)
            mask_tensor = torch.zeros((n_samples, self.seq_len), dtype=torch.float32)
        else:
            # Train/Val sets have targets in columns
            # Extract list columns and stack them
            target_arrays = []
            for col in Config.TARGET_COLS:
                # df[col] contains lists of length 68. Stack to get (N, 68)
                # Note: We assume all rows have valid data as per metadata generation
                col_data = np.vstack(df[col].values)
                target_arrays.append(col_data)

            # Stack targets along the last dimension: (N, 68, 5)
            targets_scored = np.stack(target_arrays, axis=-1)

            # Create full target tensor (N, 107, 5) padded with zeros
            full_targets = np.zeros(
                (n_samples, self.seq_len, Config.NUM_TARGETS), dtype=np.float32
            )
            full_targets[:, : self.pred_len, :] = targets_scored
            target_tensor = torch.tensor(full_targets, dtype=torch.float32)

            # Create Mask (N, 107)
            # 1.0 for positions 0-67, 0.0 for 68-106
            mask = np.zeros((n_samples, self.seq_len), dtype=np.float32)
            mask[:, : self.pred_len] = 1.0
            mask_tensor = torch.tensor(mask, dtype=torch.float32)

        # Pack into dictionary
        data_dict = {
            "ids": ids,
            "sequence": seq_tensor,
            "structure": struct_tensor,
            "loop": loop_tensor,
            "targets": target_tensor,
            "mask": mask_tensor,
        }

        # 3. Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(data_dict, self.cache_path)
        # print(f"Saved processed data to {self.cache_path}")

        return data_dict

    def __len__(self):
        return len(self.data["ids"])

    def __getitem__(self, idx):
        """
        Returns a dictionary containing inputs and targets for a single sample.
        """
        return {
            "id": self.data["ids"][idx],
            "sequence": self.data["sequence"][idx],  # (107,)
            "structure": self.data["structure"][idx],  # (107,)
            "loop": self.data["loop"][idx],  # (107,)
            "targets": self.data["targets"][idx],  # (107, 5)
            "mask": self.data["mask"][idx],  # (107,)
        }
