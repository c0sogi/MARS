import torch
from torch.utils.data import DataLoader
from library.config import Config, process_data, RNADataset


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, max_samples=None
):
    """
    Orchestrates the loading, preprocessing, and batching of RNA data.

    Args:
        batch_size (int): Number of samples per batch. Defaults to Config.BATCH_SIZE.
        load_cached_data (bool): If True, attempts to load pre-processed .npz files
                                 from the cache directory. If False or cache missing,
                                 reprocesses data.
        max_samples (int, optional): If provided, limits the dataset size to this number
                                     of samples. Useful for debugging and quick tests.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 1. Process or Load Data
    # The process_data function from library.config handles:
    # - Caching logic (checking Config.CACHE_DIR)
    # - Feature generation (One-hot Seq, Struct, Loop, Partner Identity)
    # - Target parsing and padding
    train_data = process_data(
        Config.TRAIN_CSV, mode="train", load_cached_data=load_cached_data
    )
    val_data = process_data(
        Config.VAL_CSV, mode="val", load_cached_data=load_cached_data
    )
    test_data = process_data(
        Config.TEST_CSV, mode="test", load_cached_data=load_cached_data
    )

    # 2. Handle Debugging/Subsetting
    # Slice the arrays in the dictionary if max_samples is set
    if max_samples is not None:
        for data_dict in [train_data, val_data, test_data]:
            # The dictionary contains arrays like 'inputs', 'targets', 'ids', 'partner_indices'
            # All should have the same length in the first dimension.
            current_len = len(data_dict["ids"])
            limit = min(max_samples, current_len)

            for key in data_dict.keys():
                # Ensure we are slicing a numpy array or list
                data_dict[key] = data_dict[key][:limit]

    # 3. Instantiate Datasets
    # RNADataset wraps the dictionary and handles tensor conversion and permuting
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # 4. Create Dataloaders
    # num_workers=4 is appropriate for the 12 vCPU environment
    # pin_memory=True accelerates transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to maintain consistent stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
