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


def prepare_folds(load_cached_data=True):
    """
    Prepares the training data with stratified k-fold splits.
    Implements caching to ensure deterministic splits across runs.
    """
    cache_path = os.path.join(Config.WORK_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded cached folds from {cache_path}")
            return df
        except Exception:
            pass  # Fallback to re-creation

    # 2. Create folds from scratch
    # Load train and val metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Combine them to use the full dataset for cross-validation
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    full_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    # print(f"Created and saved folds to {cache_path}")

    return full_df


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology images.
    Handles loading, center cropping (Hard Attention), and transformations.
    """

    def __init__(self, df, transform=None, data_dir=Config.INPUT_DIR, is_test=False):
        self.df = df
        self.transform = transform
        self.data_dir = data_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # The metadata file_path is relative to input dir (e.g., "train/id.tif")
        img_path = os.path.join(self.data_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for robustness, though analysis showed no missing files
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations (includes CenterCrop)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal fallback transform if none provided: CenterCrop + ToTensor
            fallback = A.Compose(
                [
                    A.CenterCrop(
                        height=Config.CENTER_CROP_SIZE, width=Config.CENTER_CROP_SIZE
                    ),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = fallback(image=image)["image"]

        # Return data based on phase
        if self.is_test:
            return image, row["id"]
        else:
            # Return label as float tensor for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32).unsqueeze(0)
            return image, label


def get_transforms(phase):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # Common normalization (ImageNet defaults)
    # Note: Config doesn't specify mean/std, so we use standard ImageNet values
    # which are appropriate for pretrained models.
    normalization = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Hard Attention: Always crop the center 48x48 first
    # This ensures we focus on the ROI before any geometric transforms
    crop = A.CenterCrop(height=Config.CENTER_CROP_SIZE, width=Config.CENTER_CROP_SIZE)

    if phase == "train":
        return A.Compose(
            [
                crop,
                # Geometric Augmentations
                A.HorizontalFlip(p=Config.AUG_H_FLIP_PROB),
                A.VerticalFlip(p=Config.AUG_V_FLIP_PROB),
                A.RandomRotate90(p=0.5),
                # Color Augmentations (Conservative: No hue/saturation)
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS,
                    contrast_limit=Config.AUG_CONTRAST,
                    p=0.5,
                ),
                normalization,
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / TTA
        # For TTA, geometric transforms are applied externally or via specific TTA pipelines.
        # Here we define the standard inference transform.
        return A.Compose([crop, normalization, ToTensorV2()])


def get_fold_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in the 5-Fold Cross-Validation.

    Args:
        fold_idx (int): The fold index (0-4) to use as validation.
        load_cached_data (bool): Whether to use cached split data.

    Returns:
        train_loader, val_loader
    """
    df = prepare_folds(load_cached_data=load_cached_data)

    # Split into train and validation for this fold
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Debugging: subset data if needed
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Create Datasets
    train_dataset = PathologyDataset(
        train_df, transform=get_transforms("train"), data_dir=Config.INPUT_DIR
    )
    val_dataset = PathologyDataset(
        val_df, transform=get_transforms("val"), data_dir=Config.INPUT_DIR
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates a DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_METADATA)

    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    test_dataset = PathologyDataset(
        test_df,
        transform=get_transforms("test"),
        data_dir=Config.INPUT_DIR,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return test_loader
