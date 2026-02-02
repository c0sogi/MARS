import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import process_data, RNADataset


class RNAProcessor:
    """
    Handles data loading, feature generation, and caching for RNA data.
    """

    def __init__(self, working_dir="./working/idea_65"):
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

    def load_and_process(self, metadata_path, mode, cache_name, load_cached_data=True):
        """
        Loads data from cache or processes it from scratch using the provided metadata CSV.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            cache_name (str): Name of the cache file (e.g., 'train_data_v1.npz').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing processed numpy arrays.
        """
        cache_path = os.path.join(self.working_dir, cache_name)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {mode} data from cache: {cache_path}")
            try:
                # Load and convert NpzFile to dict immediately to ensure data is in memory
                data = dict(np.load(cache_path, allow_pickle=True))
                return data
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

        # 2. Process from Scratch
        print(f"Processing {mode} data from scratch...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Determine if test mode based on mode string
        is_test = mode == "test"

        # Use the provided library function for consistent feature generation
        # This generates: seq, struct, loop, pid, partner_idx, and targets (if train/val)
        data = process_data(df, is_test=is_test)

        # 3. Save to Cache
        print(f"Saving {mode} data to cache: {cache_path}")
        np.savez(cache_path, **data)

        return data


def get_dataloaders(
    data_dir="./metadata",
    working_dir="./working/idea_65",
    batch_size=16,
    num_workers=2,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        data_dir (str): Directory containing metadata CSVs.
        working_dir (str): Directory for caching processed data.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use existing cache files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    processor = RNAProcessor(working_dir)

    # --- Train Data ---
    train_data = processor.load_and_process(
        metadata_path=os.path.join(data_dir, "train.csv"),
        mode="train",
        cache_name="train_data_ss_rfn_v1.npz",
        load_cached_data=load_cached_data,
    )
    train_ds = RNADataset(train_data, mode="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # --- Validation Data ---
    val_data = processor.load_and_process(
        metadata_path=os.path.join(data_dir, "val.csv"),
        mode="val",
        cache_name="val_data_ss_rfn_v1.npz",
        load_cached_data=load_cached_data,
    )
    val_ds = RNADataset(val_data, mode="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Test Data ---
    test_data = processor.load_and_process(
        metadata_path=os.path.join(data_dir, "test.csv"),
        mode="test",
        cache_name="test_data_ss_rfn_v1.npz",
        load_cached_data=load_cached_data,
    )
    test_ds = RNADataset(test_data, mode="test")

    # Dynamically attach IDs to the dataset so they can be accessed during inference
    # This is necessary because the provided RNADataset class does not store IDs by default
    if "ids" in test_data:
        test_ds.ids = test_data["ids"]

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
