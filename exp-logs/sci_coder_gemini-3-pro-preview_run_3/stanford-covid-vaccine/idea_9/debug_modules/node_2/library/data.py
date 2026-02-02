import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNADataset(Dataset):
    def __init__(self, data, is_test=False):
        """
        Args:
            data (np.ndarray): Array of shape (N, 107, 19) containing features and targets.
                               Features are indices 0-13. Targets are indices 14-18.
            is_test (bool): If True, targets are placeholders.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.is_test = is_test

        # Dimensions based on One-Hot Encoding strategy
        # Sequence(4) + Structure(3) + Loop(7) = 14
        self.feature_dim = 14
        self.target_dim = 5
        self.seq_scored = 68

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract features: (107, 14)
        features = self.data[idx, :, : self.feature_dim]

        # Extract targets: (68, 5)
        # We slice the first 68 positions which correspond to the ground truth
        targets = self.data[idx, : self.seq_scored, self.feature_dim :]

        return features, targets


def preprocess_data(df, config, is_test=False):
    """
    Converts DataFrame into a numpy array of shape (N, 107, 19).
    Channels 0-13: One-hot encoded features.
    Channels 14-18: Targets (padded to 107 for storage alignment).
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize output array
    # 14 features + 5 targets = 19 channels
    output = np.zeros((num_samples, seq_len, 19), dtype=np.float32)

    # Mappings
    seq_map = config.token2int_seq
    struct_map = config.token2int_struct
    loop_map = config.token2int_loop

    # Target columns
    target_cols = config.target_cols

    for i, row in df.iterrows():
        # --- Features ---
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Sequence (4 channels)
        for j, char in enumerate(sequence):
            if char in seq_map:
                output[i, j, seq_map[char]] = 1.0

        # 2. Structure (3 channels)
        for j, char in enumerate(structure):
            if char in struct_map:
                output[i, j, 4 + struct_map[char]] = 1.0

        # 3. Loop Type (7 channels)
        for j, char in enumerate(loop_type):
            if char in loop_map:
                output[i, j, 7 + loop_map[char]] = 1.0

        # --- Targets ---
        if not is_test:
            # Targets are provided as lists of length 68
            for k, col in enumerate(target_cols):
                val_list = row[col]
                # Assign to the first 68 positions in the corresponding channel
                # Channels start at index 14
                output[i, : len(val_list), 14 + k] = np.array(
                    val_list, dtype=np.float32
                )
        else:
            # For test, targets remain 0
            pass

    return output


def get_dataloaders(config, load_cached_data=True):
    """
    Loads data, preprocesses (or loads cache), and returns DataLoaders.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(config.seed)

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # --- Helper to handle cache logic ---
    def load_or_process(metadata_path, cache_path, is_test=False):
        data = None

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached data from {cache_path}...")
                data = np.load(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
                data = None

        # Process if not loaded
        if data is None:
            print(f"Processing data from {metadata_path}...")
            df = pd.read_parquet(metadata_path)

            # Debug mode: subset data for speed
            if config.debug:
                df = df.head(100)

            data = preprocess_data(df, config, is_test=is_test)

            print(f"Saving cache to {cache_path}...")
            np.save(cache_path, data)

        return data

    # --- Load Data ---

    # 1. Train
    # Strategy: Full-Spectrum Training. We use the full train.parquet which includes
    # all samples (both high and low SN_filter). We do not filter rows.
    train_data = load_or_process(
        config.train_metadata_path, config.train_cache_path, is_test=False
    )

    # 2. Val
    val_data = load_or_process(
        config.val_metadata_path, config.val_cache_path, is_test=False
    )

    # 3. Test
    test_data = load_or_process(
        config.test_metadata_path, config.test_cache_path, is_test=True
    )

    # --- Create Datasets ---
    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.device == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.device == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.device == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
