import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.image_ids = df["image_id"].values
        self.file_paths = df["file_path"].values

        # Pre-fetch labels for train/val modes
        if self.mode != "test":
            self.labels = df[Config.CLASS_LABELS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        # Metadata paths are relative to INPUT_DIR (e.g., "images/Train_0.jpg")
        image_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            return image, self.image_ids[idx]
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for augmentation, 'valid' or 'test' for deterministic resizing.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),  # Explicitly included per strategy
                A.ShiftScaleRotate(
                    shift_limit=Config.SCALE_LIMIT,
                    scale_limit=Config.SCALE_LIMIT,
                    rotate_limit=Config.ROTATE_LIMIT,
                    p=Config.AUG_PROB,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=Config.BRIGHTNESS_LIMIT,
                    contrast_limit=Config.CONTRAST_LIMIT,
                    p=Config.AUG_PROB,
                ),
                A.Normalize(
                    mean=Config.MEAN,
                    std=Config.STD,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=Config.MEAN,
                    std=Config.STD,
                ),
                ToTensorV2(),
            ]
        )


def calculate_class_weights(df):
    """
    Calculates class weights inversely proportional to class frequencies.
    """
    labels = df[Config.CLASS_LABELS].values
    # Assuming one-hot or soft labels, sum them up
    class_counts = np.sum(labels, axis=0)
    total_samples = len(df)
    num_classes = len(Config.CLASS_LABELS)

    # Weight = Total / (Num_Classes * Class_Count)
    weights = total_samples / (num_classes * class_counts)

    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(fold_idx=None, phase="phase1", batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        fold_idx (int, optional): The fold index for Cross-Validation (0 to N_FOLDS-1).
        phase (str): 'phase1' (CV) or 'phase2' (Full Data Training).
        batch_size (int): Batch size.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Combine train and val metadata to have the full labeled dataset
    full_train_df = pd.concat([train_meta, val_meta], ignore_index=True)

    train_df = None
    val_df = None

    # 2. Split Data based on Phase
    if phase == "phase1":
        if fold_idx is None:
            raise ValueError("fold_idx must be provided for phase1 (CV).")

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # We use 'stratify_label' which represents the dominant class
        # Iterate to find the indices for the specific fold
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["stratify_label"])
        ):
            if fold == fold_idx:
                train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
                val_df = full_train_df.iloc[val_idx].reset_index(drop=True)
                break

    elif phase == "phase2":
        # Use 100% of data for training
        train_df = full_train_df
        val_df = None  # No validation set in production training

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # 3. Calculate Class Weights (based on training split)
    class_weights = calculate_class_weights(train_df)

    # 4. Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms(data="train"), mode="train"
    )

    val_dataset = None
    if val_df is not None:
        val_dataset = AppleDataset(
            val_df, transforms=get_transforms(data="valid"), mode="val"
        )

    test_dataset = AppleDataset(
        test_meta, transforms=get_transforms(data="valid"), mode="test"
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
