import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import Config, process_dataframe, RNADataset
from library.utils import seed_everything


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    train_path=Config.TRAIN_META,
    val_path=Config.VAL_META,
    test_path=Config.TEST_META,
):
    """
    Constructs DataLoaders for Train, Validation, and Test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.
        train_path (str): Path to training metadata parquet.
        val_path (str): Path to validation metadata parquet.
        test_path (str): Path to test metadata parquet.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Load Metadata (Parquet files preserve data types)
    # These files contain the stratified splits generated in the metadata step.
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # 2. Process Data with Caching
    # process_dataframe handles:
    # - One-Hot Encoding (Sequence, Structure, Loop Type)
    # - Adjacency Matrix & Mask generation
    # - Target extraction
    # - Caching to ./working/idea_68/

    t_feat, t_pidx, t_pmask, t_targ, t_ids = process_dataframe(
        train_df, "train_data", load_cached_data=load_cached_data
    )

    v_feat, v_pidx, v_pmask, v_targ, v_ids = process_dataframe(
        val_df, "val_data", load_cached_data=load_cached_data
    )

    te_feat, te_pidx, te_pmask, te_targ, te_ids = process_dataframe(
        test_df, "test_data", load_cached_data=load_cached_data
    )

    # 3. Instantiate Datasets
    train_ds = RNADataset(t_feat, t_pidx, t_pmask, t_targ, t_ids)
    val_ds = RNADataset(v_feat, v_pidx, v_pmask, v_targ, v_ids)
    test_ds = RNADataset(te_feat, te_pidx, te_pmask, te_targ, te_ids)

    # 4. Create DataLoaders
    # pin_memory=True speeds up host-to-device transfer
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batch to maintain stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
