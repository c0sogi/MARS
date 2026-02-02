import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Feature Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns:
        indices: (L,) array where indices[i] = j if (i, j) are paired.
                 If unpaired, indices[i] = i (points to self).
        mask: (L,) array, 1.0 if paired, 0.0 if unpaired.
    """
    length = len(structure)
    indices = np.arange(length)  # Default points to self
    mask = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
    return indices, mask


def one_hot_encode(seq, mapping, depth):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    indices = [mapping[char] for char in seq]
    # Create (L, Depth)
    one_hot = np.eye(depth)[indices]
    return one_hot


class RNADataset(Dataset):
    def __init__(
        self, df, mode="train", config=None, cache_path=None, load_cached_data=True
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing raw data.
            mode (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
            cache_path (str): Path to save/load .npz cache.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.mode = mode
        self.config = config
        self.seq_scored = config.SEQ_SCORED
        self.num_targets = config.NUM_TARGETS

        # Data containers
        self.inputs = None  # (N, 107, 14)
        self.bpp_indices = None  # (N, 107)
        self.pair_masks = None  # (N, 107)
        self.targets = None  # (N, 68, 5) or None
        self.ids = None  # (N,)

        # Attempt to load cache
        if load_cached_data and cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading cached data from {cache_path}...")
                data = np.load(cache_path, allow_pickle=True)
                self.inputs = data["inputs"]
                self.bpp_indices = data["bpp_indices"]
                self.pair_masks = data["pair_masks"]
                self.ids = data["ids"]
                if "targets" in data:
                    self.targets = data["targets"]
                print("Cache loaded successfully.")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing data.")

        # Process data from scratch
        print(f"Processing {mode} data...")
        self._process_dataframe(df)

        # Save cache
        if cache_path:
            print(f"Saving cache to {cache_path}...")
            save_dict = {
                "inputs": self.inputs,
                "bpp_indices": self.bpp_indices,
                "pair_masks": self.pair_masks,
                "ids": self.ids,
            }
            if self.targets is not None:
                save_dict["targets"] = self.targets
            np.savez(cache_path, **save_dict)

    def _process_dataframe(self, df):
        n_samples = len(df)
        seq_len = self.config.SEQ_LENGTH

        # Initialize arrays
        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
        self.inputs = np.zeros((n_samples, seq_len, 14), dtype=np.float32)
        self.bpp_indices = np.zeros((n_samples, seq_len), dtype=np.int64)
        self.pair_masks = np.zeros((n_samples, seq_len), dtype=np.float32)
        self.ids = df["id"].values

        if self.mode in ["train", "val"]:
            self.targets = np.zeros(
                (n_samples, self.seq_scored, self.num_targets), dtype=np.float32
            )

        # Iterate and process
        for idx, row in df.iterrows():
            # 1. Input Features
            # Sequence (4)
            seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, 4)
            # Structure (3)
            struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, 3)
            # Loop Type (7)
            loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

            # Concatenate features
            self.inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

            # 2. Adjacency Map
            indices, mask = get_structure_adj(row["structure"])
            self.bpp_indices[idx] = indices
            self.pair_masks[idx] = mask

            # 3. Targets (only for train/val)
            if self.mode in ["train", "val"]:
                # Targets are stored as lists in the dataframe/parquet
                # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                t_react = np.array(row["reactivity"], dtype=np.float32)
                t_mg_ph10 = np.array(row["deg_Mg_pH10"], dtype=np.float32)
                t_ph10 = np.array(row["deg_pH10"], dtype=np.float32)
                t_mg_50c = np.array(row["deg_Mg_50C"], dtype=np.float32)
                t_50c = np.array(row["deg_50C"], dtype=np.float32)

                # Stack: (68, 5)
                # Ensure length matches seq_scored (68)
                # Note: Parquet lists might vary slightly if data is messy, but competition data is usually consistent.
                # We slice to seq_scored just in case.
                limit = self.seq_scored
                stack = np.stack(
                    [
                        t_react[:limit],
                        t_mg_ph10[:limit],
                        t_ph10[:limit],
                        t_mg_50c[:limit],
                        t_50c[:limit],
                    ],
                    axis=1,
                )

                self.targets[idx] = stack

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            "sequence": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "bpp_indices": torch.tensor(self.bpp_indices[idx], dtype=torch.long),
            "pair_mask": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_dataloaders(config: Config, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Load DataFrames
    # We use pandas to read parquet files generated in metadata
    df_train = pd.read_parquet(config.TRAIN_PATH)
    df_val = pd.read_parquet(config.VAL_PATH)
    df_test = pd.read_parquet(config.TEST_PATH)

    # Debug mode: subsample data
    if config.DEBUG:
        print("DEBUG MODE: Subsampling datasets...")
        df_train = df_train.iloc[:100]
        df_val = df_val.iloc[:50]
        df_test = df_test.iloc[:50]

    # Define Cache Paths
    # We append 'debug' to cache name if in debug mode to avoid overwriting full cache
    suffix = "_debug" if config.DEBUG else ""
    train_cache = os.path.join(config.CACHE_DIR, f"train_data{suffix}.npz")
    val_cache = os.path.join(config.CACHE_DIR, f"val_data{suffix}.npz")
    test_cache = os.path.join(config.CACHE_DIR, f"test_data{suffix}.npz")

    # Create Datasets
    train_dataset = RNADataset(
        df_train,
        mode="train",
        config=config,
        cache_path=train_cache,
        load_cached_data=load_cached_data,
    )

    val_dataset = RNADataset(
        df_val,
        mode="val",
        config=config,
        cache_path=val_cache,
        load_cached_data=load_cached_data,
    )

    test_dataset = RNADataset(
        df_test,
        mode="test",
        config=config,
        cache_path=test_cache,
        load_cached_data=load_cached_data,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
