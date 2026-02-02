import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def circle_crop(img, tol=7):
    """
    Crops the image to the bounding box of the non-black region (the eye).
    If the image is too dark or cropping fails, returns the original image.
    """
    if img is None:
        return img

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
    return img


class RetinopathyDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train", input_dir="./input"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # Metadata contains relative path e.g., "train_images/xxx.png"
        img_path = os.path.join(self.input_dir, row["file_path"])

        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen with verified metadata)
            # Create a black image of standard size
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Circle Crop
        image = circle_crop(image)

        # Apply Augmentations/Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode == "test":
            return image
        else:
            # Diagnosis is an integer 0-4.
            # For regression, we treat it as float32.
            label = torch.tensor(row["diagnosis"], dtype=torch.float32)
            return image, label


def get_transforms(img_size, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        img_size (int): Target image size (e.g., 512 or 384).
        mode (str): 'train' or 'val'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                # Stochastic CLAHE (p=0.5) as per idea
                A.CLAHE(clip_limit=4.0, p=Config.CLAHE_PROB),
                # Geometric Augmentations
                A.HorizontalFlip(p=Config.AUG_FLIP_PROB),
                A.VerticalFlip(p=Config.AUG_FLIP_PROB),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=Config.AUG_ROTATION,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def load_dataframe(csv_path, cache_name, load_cached_data=True):
    """
    Loads a dataframe with caching logic.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, fall back to creating it
            pass

    # Load from source CSV
    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass  # Non-critical failure

    return df


def get_dataloaders(img_size, batch_size, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        img_size (int): Target resolution for images.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = load_dataframe(Config.TRAIN_CSV, "train_df", load_cached_data)
    val_df = load_dataframe(Config.VAL_CSV, "val_df", load_cached_data)
    test_df = load_dataframe(Config.TEST_CSV, "test_df", load_cached_data)

    # Debug mode: subset data
    if Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(20)

    # Define Transforms
    train_transforms = get_transforms(img_size, mode="train")
    val_transforms = get_transforms(img_size, mode="val")

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_df, transforms=train_transforms, mode="train", input_dir=Config.INPUT_DIR
    )

    val_dataset = RetinopathyDataset(
        val_df, transforms=val_transforms, mode="val", input_dir=Config.INPUT_DIR
    )

    test_dataset = RetinopathyDataset(
        test_df, transforms=val_transforms, mode="test", input_dir=Config.INPUT_DIR
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
