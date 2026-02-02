import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything


class RetinopathyDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy Detection.
    Handles image loading, aspect-ratio preserving padding, resizing, and transformations.
    """

    def __init__(self, df, transforms=None, img_size=Config.IMG_SIZE):
        self.df = df
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata paths are relative to input dir (e.g., "train_images/id.png")
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(file_path)

        # Handle potential missing/corrupt images gracefully
        if image is None:
            # Return a black image of target size
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Pad to Square (Preserve Aspect Ratio)
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if h != max_dim or w != max_dim:
            top = (max_dim - h) // 2
            bottom = max_dim - h - top
            left = (max_dim - w) // 2
            right = max_dim - w - left

            # Pad with black borders (value=0)
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        # 2. Resize to Target Resolution
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(
                image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )

        # 3. Apply Transforms (Augmentations + Normalize + ToTensor)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback (Normalize to 0-1 and convert to tensor)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 4. Get Label
        # For regression, we use float targets
        if "diagnosis" in row:
            label = torch.tensor(row["diagnosis"], dtype=torch.float)
        else:
            # Placeholder for test set if diagnosis is missing
            label = torch.tensor(0.0, dtype=torch.float)

        return image, label


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Geometric Augmentations (Safe for medical images)
                A.HorizontalFlip(p=0.5),
                # Color/Intensity Augmentations
                A.RandomBrightnessContrast(p=0.2),
                # Normalization and Tensor Conversion
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only Normalize and ToTensor
        return A.Compose([A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()])


def get_dataloaders(
    debug=Config.DEBUG,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, use a small subset of data.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Flag to satisfy interface requirements.
                                 Since metadata is pre-generated CSVs, we load them directly.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_df, transforms=get_transforms("train"), img_size=Config.IMG_SIZE
    )

    val_dataset = RetinopathyDataset(
        val_df, transforms=get_transforms("val"), img_size=Config.IMG_SIZE
    )

    test_dataset = RetinopathyDataset(
        test_df, transforms=get_transforms("test"), img_size=Config.IMG_SIZE
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
