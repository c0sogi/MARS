import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def load_metadata():
    """
    Loads the train, validation, and test metadata dataframes from the configured paths.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    return train_df, val_df, test_df


def circle_crop(img, tol=7):
    """
    Crops the image to the bounding box of the non-black pixels (eye fundus).
    This removes the uninformative black borders common in fundus photography.

    Args:
        img (numpy.ndarray): Input image.
        tol (int): Tolerance for considering a pixel as 'black'.

    Returns:
        numpy.ndarray: Cropped image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            # Image is too dark or empty, return original
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def get_transforms(image_size, mode="train"):
    """
    Returns the albumentations transforms for the given mode and image size.

    Args:
        image_size (int): The target height and width for resizing.
        mode (str): 'train' for augmentation, 'val'/'test' for deterministic transforms.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=Config.ROTATION_LIMIT, p=0.5),
                # Stochastic CLAHE to handle contrast variations
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=Config.CLAHE_PROB),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


class RetinopathyDataset(Dataset):
    """
    Dataset class for Diabetic Retinopathy images.
    Handles loading, circle cropping, and applying transforms.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'diagnosis' (for train/val).
            transform (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images
            # Create a black image of standard size to prevent crash
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply circle crop to remove black borders
        image = circle_crop(image)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            # For test, return image and dummy target (0)
            # The caller will use the DataFrame order to map predictions to IDs
            return image, 0
        else:
            label = row["diagnosis"]
            # Return float for regression (MSE Loss)
            return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(train_df, val_df, test_df, image_size, batch_size):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        image_size (int): Image resolution for this phase.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transforms = get_transforms(image_size, mode="train")
    val_transforms = get_transforms(image_size, mode="val")

    train_ds = RetinopathyDataset(train_df, transform=train_transforms, mode="train")
    val_ds = RetinopathyDataset(val_df, transform=val_transforms, mode="val")
    test_ds = RetinopathyDataset(test_df, transform=val_transforms, mode="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
