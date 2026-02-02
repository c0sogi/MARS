import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


class CatDogDataset(Dataset):
    """
    Custom Dataset for loading Dog vs Cat images.
    """

    def __init__(self, df, root_dir, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            root_dir (str): Root directory for images (e.g., ./input).
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # filepath in metadata is relative to input dir (e.g., "train/cat.0.jpg")
        img_path = os.path.join(self.root_dir, row["filepath"])

        try:
            # Load image and convert to RGB
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            # Return a blank image to prevent crashing, though this shouldn't happen with clean data
            image = Image.new("RGB", (256, 256))

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # For test set, return image and id
            return image, row["id"]
        else:
            # For train/val, return image and label
            # Ensure label is float for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label


def get_transforms(cfg, mode="train"):
    """
    Generates the transformation pipeline based on model config and mode.

    Args:
        cfg (dict): Model configuration dictionary containing 'img_size'.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    img_size = cfg["img_size"]

    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        # Augmentation pipeline for training
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (img_size, img_size),
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Deterministic pipeline for validation/test
        # Using Resize to fixed dimensions as per "Static Resolution" strategy
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transform


def create_kfold_loaders(model_key):
    """
    Creates Stratified K-Fold DataLoaders for the specified model.
    Merges original train and val metadata to utilize 100% of data.

    Args:
        model_key (str): Key identifying the model in Config.MODELS.

    Returns:
        list: A list of tuples (train_loader, val_loader) for each fold.
    """
    if model_key not in Config.MODELS:
        raise ValueError(f"Model key '{model_key}' not found in Config.MODELS")

    model_cfg = Config.MODELS[model_key]
    batch_size = model_cfg["batch_size"]

    # 1. Load and Merge Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    train_df_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df_part = pd.read_csv(Config.VAL_METADATA_PATH)

    full_df = pd.concat([train_df_part, val_df_part], ignore_index=True)

    # Handle DEBUG mode
    if Config.DEBUG:
        full_df = full_df.sample(
            n=Config.DEBUG_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)
        logger.info(
            f"DEBUG Mode: Using {len(full_df)} samples for training/validation."
        )

    # 2. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    folds = []
    X = np.zeros(len(full_df))  # Dummy features
    y = full_df["label"].values

    logger.info(
        f"Preparing {Config.N_FOLDS} folds for {model_key} (Img Size: {model_cfg['img_size']})..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Get Transforms
        train_transform = get_transforms(model_cfg, mode="train")
        val_transform = get_transforms(model_cfg, mode="val")

        # Create Datasets
        train_dataset = CatDogDataset(
            fold_train_df, Config.INPUT_DIR, transform=train_transform, mode="train"
        )
        val_dataset = CatDogDataset(
            fold_val_df, Config.INPUT_DIR, transform=val_transform, mode="val"
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,  # Drop last incomplete batch to maintain stable statistics
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        folds.append((train_loader, val_loader))

    return folds


def create_test_loader(model_key):
    """
    Creates the DataLoader for the Test set.

    Args:
        model_key (str): Key identifying the model in Config.MODELS.

    Returns:
        DataLoader: The test data loader.
    """
    if model_key not in Config.MODELS:
        raise ValueError(f"Model key '{model_key}' not found in Config.MODELS")

    model_cfg = Config.MODELS[model_key]
    batch_size = model_cfg["batch_size"]

    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError("Test metadata file not found.")

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if Config.DEBUG:
        test_df = test_df.sample(
            n=Config.DEBUG_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)
        logger.info(f"DEBUG Mode: Using {len(test_df)} samples for testing.")

    # Use 'test' mode transform (same as val)
    transform = get_transforms(model_cfg, mode="test")

    dataset = CatDogDataset(test_df, Config.INPUT_DIR, transform=transform, mode="test")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
