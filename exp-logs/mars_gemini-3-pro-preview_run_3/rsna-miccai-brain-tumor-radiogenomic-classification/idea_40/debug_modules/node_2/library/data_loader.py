import os
import torch
from torch.utils.data import DataLoader
from library.utils import load_data_and_cache, SiameseDataset

# Alias the dataset class as requested in the description
SiameseMRIDataset = SiameseDataset


def get_dataloaders(
    train_meta_path="./metadata/train.parquet",
    val_meta_path="./metadata/val.parquet",
    test_meta_path="./metadata/test.parquet",
    batch_size=16,
    num_workers=4,
    load_cached_data=True,
    cache_dir="./working/idea_40/",
):
    """
    Orchestrates the creation of DataLoaders for Train, Validation, and Test sets.
    Leverages the caching mechanism to store processed Even/Odd streams.
    """

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    loaders = {}

    # 1. Training Loader
    if os.path.exists(train_meta_path):
        X_even, X_odd, y, _ = load_data_and_cache(
            train_meta_path,
            cache_dir=cache_dir,
            load_cached_data=load_cached_data,
            dataset_name="train",
        )
        train_dataset = SiameseMRIDataset(X_even, X_odd, y)
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        loaders["train"] = None

    # 2. Validation Loader
    if os.path.exists(val_meta_path):
        X_even, X_odd, y, _ = load_data_and_cache(
            val_meta_path,
            cache_dir=cache_dir,
            load_cached_data=load_cached_data,
            dataset_name="val",
        )
        val_dataset = SiameseMRIDataset(X_even, X_odd, y)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        loaders["val"] = None

    # 3. Test Loader
    if os.path.exists(test_meta_path):
        X_even, X_odd, y, ids = load_data_and_cache(
            test_meta_path,
            cache_dir=cache_dir,
            load_cached_data=load_cached_data,
            dataset_name="test",
        )
        # y will be None for test set usually, but SiameseDataset handles it
        test_dataset = SiameseMRIDataset(X_even, X_odd, y)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        # Return IDs separately or attach to loader if needed,
        # but standard is to return loaders. IDs are cached if needed for submission.
    else:
        loaders["test"] = None

    return loaders["train"], loaders["val"], loaders["test"]
