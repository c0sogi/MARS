import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    Loads images and generates binary targets for multi-label decomposition.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose = None,
        mode: str = "train",
        input_dir: str = "./input",
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transform (A.Compose): Albumentations transformations.
            mode (str): 'train' or 'test'. If 'train', returns targets.
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "images/Train_0.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen based on EDA)
            # Create a black image of expected size to prevent crash
            image = np.zeros((480, 480, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and targets
        if self.mode == "train":
            # Targets: [is_rust, is_scab]
            # Derived from processed columns in the dataframe
            targets = torch.tensor(
                [row["is_rust"], row["is_scab"]], dtype=torch.float32
            )
            return image, targets
        else:
            return image, row["image_id"]


def get_transforms(mode: str, cfg: Config) -> A.Compose:
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train' or 'valid'/'test'.
        cfg (Config): Configuration object.
    """
    transforms = []

    # Resize to native resolution of EfficientNetV2-L
    transforms.append(A.Resize(height=cfg.img_size, width=cfg.img_size))

    if mode == "train":
        # Augmentations for training
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.VerticalFlip(p=0.5))

        # CoarseDropout for distributed feature learning
        cd_params = cfg.coarse_dropout_params
        transforms.append(
            A.CoarseDropout(
                max_holes=cd_params["max_holes"],
                max_height=cd_params["max_height"],
                max_width=cd_params["max_width"],
                min_holes=cd_params["min_holes"],
                min_height=cd_params["min_height"],
                min_width=cd_params["min_width"],
                fill_value=cd_params["fill_value"],
                p=cd_params["p"],
            )
        )

    # Normalize and Convert to Tensor
    # Using ImageNet mean/std as standard for transfer learning
    transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def _process_data(cfg: Config, split: str, load_cached_data: bool) -> pd.DataFrame:
    """
    Internal function to load, process, and cache metadata.

    Args:
        cfg (Config): Configuration object.
        split (str): 'train' (combines train+val metadata) or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    cache_path = cfg.get_cache_path(f"processed_{split}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    if split == "train":
        # Load provided train and val metadata
        train_df = pd.read_csv(cfg.train_metadata_path)
        val_df = pd.read_csv(cfg.val_metadata_path)

        # Combine them for 5-Fold CV re-splitting
        df = pd.concat([train_df, val_df], ignore_index=True)

        # Generate Binary Targets for Multi-Label Decomposition
        # Rust = 1 if 'rust' column is 1 OR 'multiple_diseases' is 1
        # Scab = 1 if 'scab' column is 1 OR 'multiple_diseases' is 1
        # Assuming columns are one-hot encoded floats/ints
        df["is_rust"] = df[["rust", "multiple_diseases"]].max(axis=1).astype(float)
        df["is_scab"] = df[["scab", "multiple_diseases"]].max(axis=1).astype(float)

    else:  # test
        df = pd.read_csv(cfg.test_metadata_path)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_loaders(fold: int, cfg: Config, load_cached_data: bool = True):
    """
    Creates DataLoaders for a specific fold in 5-Fold CV.

    Args:
        fold (int): Fold index (0 to n_folds-1).
        cfg (Config): Configuration object.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        train_loader, val_loader
    """
    # Load combined training data
    df = _process_data(cfg, "train", load_cached_data)

    # Debug Mode: Sample subset
    if cfg.debug:
        df = df.sample(
            n=min(len(df), cfg.debug_sample_size), random_state=cfg.seed
        ).reset_index(drop=True)

    # Stratified K-Fold Split
    # We use 'stratify_label' which exists in the metadata
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    # Get indices for the requested fold
    # We iterate to find the specific fold indices
    train_idx, val_idx = list(skf.split(df, df["stratify_label"]))[fold]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df,
        transform=get_transforms("train", cfg),
        mode="train",
        input_dir=cfg.input_dir,
    )
    val_dataset = AppleDataset(
        val_df,
        transform=get_transforms("valid", cfg),
        mode="train",
        input_dir=cfg.input_dir,
    )

    # Worker Init Function for reproducibility
    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Drop incomplete batches for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_test_loader(cfg: Config, load_cached_data: bool = True):
    """
    Creates DataLoader for the test set.

    Args:
        cfg (Config): Configuration object.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        test_loader
    """
    df = _process_data(cfg, "test", load_cached_data)

    test_dataset = AppleDataset(
        df,
        transform=get_transforms(
            "valid", cfg
        ),  # Use valid transforms (no augmentations)
        mode="test",
        input_dir=cfg.input_dir,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return test_loader


def get_class_weights(fold: int, cfg: Config, load_cached_data: bool = True):
    """
    Calculates positive class weights for BCEWithLogitsLoss for a specific fold.
    Weight = Number of Negatives / Number of Positives.

    Args:
        fold (int): Fold index.
        cfg (Config): Configuration object.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        torch.Tensor: Weights for [is_rust, is_scab]
    """
    # Load data and replicate split logic
    df = _process_data(cfg, "train", load_cached_data)

    if cfg.debug:
        df = df.sample(
            n=min(len(df), cfg.debug_sample_size), random_state=cfg.seed
        ).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    train_idx, _ = list(skf.split(df, df["stratify_label"]))[fold]
    train_df = df.iloc[train_idx]

    # Calculate weights
    rust_pos = train_df["is_rust"].sum()
    rust_neg = len(train_df) - rust_pos
    rust_weight = rust_neg / (rust_pos + 1e-6)

    scab_pos = train_df["is_scab"].sum()
    scab_neg = len(train_df) - scab_pos
    scab_weight = scab_neg / (scab_pos + 1e-6)

    return torch.tensor([rust_weight, scab_weight], dtype=torch.float32)
