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


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    Reads images from disk and applies transformations.
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.file_paths = df["file_path"].values

        # Determine if targets are available (Train/Val) or not (Test)
        self.targets = None
        if all(col in df.columns for col in Config.TARGET_COLS):
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return image and label (if exists)
        if self.targets is not None:
            label = torch.tensor(self.targets[idx])
            return image, label
        else:
            # Return image and empty tensor for test inference
            return image, torch.tensor([])


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for augmentation, 'valid' or 'test' for resizing/norm only.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                # Strategic Retention: VerticalFlip included as per Lesson 30 for rotationally invariant leaves
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def get_loaders(fold_idx=0, phase="calibration", seed=42):
    """
    Creates DataLoaders for the specified phase.

    Args:
        fold_idx (int): Index of the fold for calibration (0 to N_FOLDS-1).
        phase (str): 'calibration', 'production', or 'test'.
        seed (int): Seed for reproducibility. Note: Global seed should be set before calling this
                    if phase is production to ensure shuffle randomness is controlled.

    Returns:
        tuple: (train_loader, val_loader) or test_loader
    """

    # ==========================
    # TEST PHASE
    # ==========================
    if phase == "test":
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = AppleDataset(df_test, transforms=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return test_loader

    # ==========================
    # LOAD & COMBINE DATA
    # ==========================
    # Load metadata parts
    df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine to recover full training set (100% data)
    df_full = pd.concat([df_train_part, df_val_part], ignore_index=True)

    # Ensure stratify label exists for splitting
    if "stratify_label" not in df_full.columns:
        df_full["stratify_label"] = df_full[Config.TARGET_COLS].idxmax(axis=1)

    # ==========================
    # PRODUCTION PHASE
    # ==========================
    if phase == "production":
        # Train on 100% of data
        train_dataset = AppleDataset(df_full, transforms=get_transforms("train"))

        # Shuffle is True. Randomness is controlled by global seed set in training script.
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return train_loader, None

    # ==========================
    # CALIBRATION PHASE
    # ==========================
    elif phase == "calibration":
        # Stratified K-Fold Split
        # We use a fixed seed (42) for the split to ensure folds are consistent
        # regardless of the training seed used for initialization.
        skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)

        splits = list(skf.split(df_full, df_full["stratify_label"]))

        if fold_idx < 0 or fold_idx >= Config.N_FOLDS:
            raise ValueError(
                f"Fold index {fold_idx} out of range (0-{Config.N_FOLDS-1})"
            )

        train_idx, val_idx = splits[fold_idx]

        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)

        train_dataset = AppleDataset(df_train, transforms=get_transforms("train"))
        val_dataset = AppleDataset(df_val, transforms=get_transforms("valid"))

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown phase: {phase}")
