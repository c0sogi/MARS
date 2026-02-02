import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def crop_image_from_gray(img, tol=7):
    """
    Crops the image to the non-black area (fundus circle) to remove uninformative borders.
    This is known as 'Ben Graham's Method' in DR competitions.

    Args:
        img (numpy.ndarray): Input image.
        tol (int): Tolerance for thresholding.

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
            # Image is too dark, return original
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def get_transforms(phase: str):
    """
    Returns the Albumentations transformations for the specific phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class RetinopathyDataset(Dataset):
    def __init__(self, csv_path, transforms=None, mode="train"):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.mode = mode

        # Debugging: subset data if Config.debug is True
        if Config.debug:
            self.df = self.df.head(Config.debug_sample_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input_dir (e.g., "train_images/id.png")
        img_path = os.path.join(Config.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (create black image)
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing: Removed crop_image_from_gray to avoid variable geometric distortion (Cite solution_lesson_node_00009)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Handle Targets
        if self.mode in ["train", "val"]:
            label = row["diagnosis"]

            # Ordinal Regression Target Generation
            # We use K-1 output units for K classes.
            # Target vector size is num_ordinal_units (4).
            # If label is k, the first k units are 1, rest are 0.
            # Label 0: [0, 0, 0, 0]
            # Label 1: [1, 0, 0, 0]
            # Label 2: [1, 1, 0, 0]
            # Label 3: [1, 1, 1, 0]
            # Label 4: [1, 1, 1, 1]
            target = np.zeros(Config.num_ordinal_units, dtype=np.float32)
            if label > 0:
                target[:label] = 1.0

            return image_tensor, torch.tensor(target, dtype=torch.float32)

        else:
            # Test mode: return image and ID for inference
            return image_tensor, row["id_code"]


def create_dataloaders():
    """
    Creates DataLoaders for train, validation, and test sets using paths from Config.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Train Dataset
    train_dataset = RetinopathyDataset(
        csv_path=Config.train_csv_path, transforms=get_transforms("train"), mode="train"
    )

    # Validation Dataset
    val_dataset = RetinopathyDataset(
        csv_path=Config.val_csv_path, transforms=get_transforms("val"), mode="val"
    )

    # Test Dataset
    test_dataset = RetinopathyDataset(
        csv_path=Config.test_csv_path, transforms=get_transforms("test"), mode="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
