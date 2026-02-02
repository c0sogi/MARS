import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class PetDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    Loads images, processes metadata features, and returns tensors.
    """

    def __init__(
        self, df: pd.DataFrame, transforms: A.Compose = None, mode: str = "train"
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            transforms (A.Compose): Albumentations transforms pipeline.
            mode (str): 'train', 'valid', or 'test'. Determines if targets are returned.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Metadata features columns (12 binary features)
        self.meta_features = [
            "Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

        # Pre-extract data to arrays for faster access
        # Ensure file_paths are strings
        self.file_paths = self.df["file_path"].values.astype(str)
        self.dense_features = self.df[self.meta_features].values.astype(np.float32)
        self.ids = self.df["Id"].values

        if self.mode in ["train", "valid"]:
            # Pawpularity is the target
            self.targets = self.df["Pawpularity"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        # Construct full image path
        # file_path in metadata is relative (e.g., "train/xxx.jpg")
        img_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Safety check for missing or corrupt images
        if image is None:
            # Return a blank image if loading fails to prevent crash
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transforms provided: Resize and ToTensor
            transform = A.Compose(
                [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()]
            )
            image = transform(image=image)["image"]

        # Prepare dense features
        dense = torch.tensor(self.dense_features[idx], dtype=torch.float32)

        sample = {"image": image, "dense_features": dense, "id": self.ids[idx]}

        # Add target for training/validation
        if self.mode in ["train", "valid"]:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["target"] = target

        return sample


def get_transforms(mode: str = "train", img_size: int = Config.IMG_SIZE) -> A.Compose:
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train', 'valid', or 'test'.
        img_size (int): Target image size.

    Returns:
        A.Compose: The transform pipeline.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                # Moderate augmentation: Random Crop and Flip
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.85, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(),
            ]
        )
    else:
        # Deterministic transforms for validation and test
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(),
            ]
        )


def load_dataset_dfs(debug: bool = Config.DEBUG):
    """
    Loads the train, validation, and test metadata DataFrames.

    Args:
        debug (bool): If True, subsets the data for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    if debug:
        print(f"Debug mode: Loading subset of {Config.DEBUG_SUBSET_SIZE} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    return train_df, val_df, test_df


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Transforms
    train_tf = get_transforms(mode="train", img_size=Config.IMG_SIZE)
    valid_tf = get_transforms(mode="valid", img_size=Config.IMG_SIZE)

    # Datasets
    # Note: Test set uses 'valid' transforms (deterministic) but mode='test'
    train_ds = PetDataset(train_df, transforms=train_tf, mode="train")
    val_ds = PetDataset(val_df, transforms=valid_tf, mode="valid")
    test_ds = PetDataset(test_df, transforms=valid_tf, mode="test")

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
