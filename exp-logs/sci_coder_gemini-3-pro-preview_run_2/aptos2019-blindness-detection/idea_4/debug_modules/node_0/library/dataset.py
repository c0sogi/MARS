import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image.

    Args:
        img (numpy.ndarray): The input image (H, W, C) or (H, W).
        tol (int): Tolerance for black pixel thresholding.

    Returns:
        numpy.ndarray: The cropped image.
    """
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


def get_transforms(image_size, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        image_size (int): Target spatial dimension (e.g., 512 or 384).
        mode (str): 'train' or 'valid'/'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Resize is handled after cropping in the dataset loop,
                # but we ensure it here as part of the pipeline logic
                A.Resize(height=image_size, width=image_size),
                # Photometric Augmentations
                A.CLAHE(p=Config.CLAHE_PROB),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=Config.ROTATION_DEGREES, p=0.5),
                # RandomResizedCrop can be beneficial for scale invariance
                A.RandomResizedCrop(
                    height=image_size,
                    width=image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
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
    PyTorch Dataset for Diabetic Retinopathy prediction.
    """

    def __init__(self, csv_path, transform=None, mode="train"):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.csv_path = csv_path
        self.transform = transform
        self.mode = mode

        # Load Data
        self.df = self._load_data()

    def _load_data(self):
        """
        Loads the dataframe.
        Note: Metadata is already generated in ./metadata, so we load directly.
        If complex processing were needed, we would implement caching here.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Construct full file path
        # Metadata contains relative path 'train_images/xxxx.png'
        # Config.INPUT_DIR is './input'
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (should be caught by metadata verification, but for safety)
            # Create a black image of standard size
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing: Remove black borders (Circle Crop logic)
        image = crop_image_from_gray(image)

        # Apply Augmentations/Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            # Regression target: float
            label = torch.tensor(row["diagnosis"], dtype=torch.float)
            return image, label
        else:
            # Test mode: return image and id_code for submission
            return image, row["id_code"]
