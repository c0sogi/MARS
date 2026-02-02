import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from library.config import Config, AppleDataset, get_transforms

# Define cache directory for fold splits
CACHE_DIR = "./working/idea_10/"


def get_folds(df, n_folds=5, seed=42, load_cached_data=True):
    """
    Assigns folds to the dataframe using Stratified K-Fold.
    Implements caching to store/retrieve fold assignments from disk.

    Args:
        df (pd.DataFrame): Input dataframe containing metadata.
        n_folds (int): Number of folds.
        seed (int): Random seed.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe with a new 'fold' column.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "folds.parquet")

    # Logic Flow 1: Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            # Basic integrity check: ensure length matches
            if len(cached_df) == len(df):
                return cached_df
        except Exception:
            # Fallback to computation if load fails
            pass

    # Logic Flow 2: Compute folds from scratch
    df = df.copy()

    # Ensure we have a label for stratification
    # Metadata guarantees 'stratify_label', but we double check or derive it
    if "stratify_label" not in df.columns:
        # Check if target columns exist to derive label
        if set(Config.target_cols).issubset(df.columns):
            df["stratify_label"] = df[Config.target_cols].idxmax(axis=1)
        else:
            # Fallback for test set or incomplete data (should not happen for training)
            df["stratify_label"] = 0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["stratify_label"])):
        df.loc[val_idx, "fold"] = fold

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_loaders(
    fold,
    img_size,
    batch_size=Config.batch_size,
    n_folds=Config.n_folds,
    seed=Config.seed,
    num_workers=Config.num_workers,
    load_cached_data=True,
    debug=False,
):
    """
    Creates PyTorch DataLoaders for training and validation for a specific fold.

    Args:
        fold (int): The fold index to use for validation (0 to n_folds-1).
        img_size (int): Image resolution (e.g., 384 or 224).
        batch_size (int): Batch size.
        n_folds (int): Total number of folds.
        seed (int): Random seed.
        num_workers (int): Number of DataLoader workers.
        load_cached_data (bool): Whether to use cached fold splits.
        debug (bool): If True, subsamples data for quick debugging.

    Returns:
        tuple: (train_loader, valid_loader)
    """
    # Load Metadata
    train_path = os.path.join(Config.metadata_dir, "train.csv")
    val_path = os.path.join(Config.metadata_dir, "val.csv")

    train_meta = pd.read_csv(train_path)
    val_meta = pd.read_csv(val_path)

    # Combine provided splits to perform our own Stratified K-Fold
    full_df = pd.concat([train_meta, val_meta]).reset_index(drop=True)

    # Get Folds (with caching)
    full_df = get_folds(
        full_df, n_folds=n_folds, seed=seed, load_cached_data=load_cached_data
    )

    # Split into Train and Validation based on fold index
    train_df = full_df[full_df["fold"] != fold].reset_index(drop=True)
    valid_df = full_df[full_df["fold"] == fold].reset_index(drop=True)

    # Debug Subsampling
    if debug:
        train_df = train_df.head(batch_size * 2)
        valid_df = valid_df.head(batch_size * 2)

    # Create Datasets
    # Note: Config.images_dir is passed but AppleDataset uses Config.input_dir internally
    # consistent with the provided library implementation.
    train_dataset = AppleDataset(
        train_df, Config.images_dir, transform=get_transforms("train", img_size)
    )
    valid_dataset = AppleDataset(
        valid_df, Config.images_dir, transform=get_transforms("valid", img_size)
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_test_loader(
    img_size, batch_size=Config.batch_size, num_workers=Config.num_workers, tta=False
):
    """
    Creates a PyTorch DataLoader for the test set.

    Args:
        img_size (int): Image resolution.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        tta (bool): If True, applies TTA specific transforms (e.g., Flip).
                    If False, applies standard validation transforms.

    Returns:
        DataLoader: The test data loader.
    """
    test_path = os.path.join(Config.metadata_dir, "test.csv")
    test_df = pd.read_csv(test_path)

    # Determine which transform to use
    transform_name = "tta_flip" if tta else "valid"

    test_dataset = AppleDataset(
        test_df, Config.images_dir, transform=get_transforms(transform_name, img_size)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
