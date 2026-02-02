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
    Custom Dataset for Digital Pathology images.
    Loads images from disk, applies Hard Attention crop, and specific augmentations.
    """

    def __init__(self, df, transforms=None, root_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, [label]).
            transforms (albumentations.Compose): Transformations to apply.
            root_dir (str): Root directory for image files.
        """
        self.df = df
        self.transforms = transforms
        self.root_dir = root_dir

        # Check if label column exists
        self.has_label = "label" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train/{id}.tif"
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        # cv2 loads in BGR, need to convert to RGB
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for broken images, though metadata check should prevent this
            # Return a black image of expected size to prevent crashing
            image = np.zeros(
                (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations (includes CenterCrop 48x48)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Handle label
        if self.has_label:
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # For test set, return dummy label or just ID tracking in outer loop
            # Returning -1 as placeholder
            return image, torch.tensor(-1, dtype=torch.float32)


def get_transforms(split):
    """
    Returns the augmentation pipeline for a specific data split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms_list = []

    # 1. Hard Attention: Center Crop to 48x48
    # This is structural and applied to all splits
    transforms_list.append(
        A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE)
    )

    if split == "train":
        # 2. Geometric Augmentations (Exploiting symmetry)
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

        # 3. Stain-Preserving Color Augmentation
        # Strictly limited to mild brightness/contrast. No Hue/Saturation.
        transforms_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        )

    # 4. Normalization and Tensor Conversion
    transforms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])

    return A.Compose(transforms_list)


def get_dataloaders(
    train_path=Config.TRAIN_METADATA_PATH,
    val_path=Config.VAL_METADATA_PATH,
    test_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=False,
    sample_size=1000,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_path (str): Path to train metadata CSV.
        val_path (str): Path to validation metadata CSV.
        test_path (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsamples the dataset for debugging.
        sample_size (int): Number of samples to use if debug is True.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    loaders = {}

    # Helper to load and prepare dataset
    def create_loader(csv_path, split, shuffle):
        if not os.path.exists(csv_path):
            print(
                f"Warning: Metadata file {csv_path} not found. Skipping {split} loader."
            )
            return None

        df = pd.read_csv(csv_path)

        if debug:
            df = df.sample(
                n=min(len(df), sample_size), random_state=Config.SEED
            ).reset_index(drop=True)

        dataset = PathologyDataset(
            df=df, transforms=get_transforms(split), root_dir=Config.INPUT_DIR
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if Config.DEVICE == "cuda" else False,
            drop_last=(
                split == "train"
            ),  # Drop last incomplete batch only for training
        )
        return loader

    # Create loaders
    loaders["train"] = create_loader(train_path, "train", shuffle=True)
    loaders["val"] = create_loader(val_path, "val", shuffle=False)
    loaders["test"] = create_loader(test_path, "test", shuffle=False)

    return loaders
