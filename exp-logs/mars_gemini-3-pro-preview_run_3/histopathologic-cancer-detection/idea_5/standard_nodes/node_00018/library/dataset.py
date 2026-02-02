import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class PathologyDataset(Dataset):
    """
    Custom Dataset for loading Digital Pathology images.
    Handles reading images from disk, applying crops/augmentations, and returning tensors.
    """

    def __init__(self, df, root_dir, transform=None, return_id=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, label, file_path).
            root_dir (str): Root directory where images are stored.
            transform (albumentations.Compose): Transformations to apply.
            return_id (bool): If True, returns (image, label, id). Useful for inference.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., "train/xxxx.tif")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing/corrupt images (though analysis showed 0 missing)
            # Create a black image of the original raw size (96x96)
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations (Crop, Augment, Normalize)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get Label
        # For test set, label might be 0 (placeholder), but we return it for consistency
        label = row["label"] if "label" in row else 0
        label = torch.tensor(label, dtype=torch.float32)

        if self.return_id:
            return image, label, row["id"]

        return image, label


def get_transforms(data="train"):
    """
    Creates the Albumentations transform pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    # The task requires predicting the center 32x32 region.
    # We use a 64x64 crop to provide context (Config.IMG_SIZE).
    crop_size = Config.IMG_SIZE

    transforms = []

    # 1. Deterministic Center Crop for all splits
    # Raw images are 96x96, we crop to 64x64
    transforms.append(A.CenterCrop(height=crop_size, width=crop_size))

    # 2. Geometric Augmentations (Train only)
    if data == "train":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    # 3. Normalization and Tensor Conversion
    # Using standard ImageNet mean/std
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def load_metadata():
    """
    Loads train, validation, and test metadata DataFrames from CSVs defined in Config.
    """
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    return train_df, val_df, test_df


def get_dataloaders(
    train_df=None, val_df=None, test_df=None, batch_size=Config.BATCH_SIZE
):
    """
    Creates PyTorch DataLoaders for the provided DataFrames.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        batch_size (int): Batch size.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders if corresponding dfs provided.
    """
    loaders = {}

    # Configure common DataLoader args
    loader_args = {
        "batch_size": batch_size,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,  # Faster transfer to GPU
    }

    # Train Loader
    if train_df is not None:
        train_ds = PathologyDataset(
            df=train_df,
            root_dir=Config.INPUT_DIR,
            transform=get_transforms(data="train"),
            return_id=False,
        )
        loaders["train"] = DataLoader(
            train_ds,
            shuffle=True,
            drop_last=True,  # Drop incomplete batch for BatchNorm stability
            **loader_args
        )

    # Validation Loader
    if val_df is not None:
        val_ds = PathologyDataset(
            df=val_df,
            root_dir=Config.INPUT_DIR,
            transform=get_transforms(data="valid"),
            return_id=False,
        )
        loaders["val"] = DataLoader(
            val_ds, shuffle=False, drop_last=False, **loader_args
        )

    # Test Loader
    if test_df is not None:
        test_ds = PathologyDataset(
            df=test_df,
            root_dir=Config.INPUT_DIR,
            transform=get_transforms(data="test"),
            return_id=True,  # Return ID for submission file creation
        )
        loaders["test"] = DataLoader(
            test_ds, shuffle=False, drop_last=False, **loader_args
        )

    return loaders
