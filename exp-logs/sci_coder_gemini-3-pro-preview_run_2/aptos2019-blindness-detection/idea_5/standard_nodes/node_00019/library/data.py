import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


def circle_crop(img, tol=7):
    """
    Removes the uninformative black borders from fundus images.

    Args:
        img (numpy.ndarray): Input image (H, W, C).
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
        if check_shape == 0:  # Image is too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def get_transforms(phase: str, img_size: int):
    """
    Generates the Albumentations transform pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.
        img_size (int): Target resolution for resizing.

    Returns:
        albumentations.Compose: Transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.CLAHE(p=CFG.clahe_prob),
                A.HorizontalFlip(p=CFG.flip_prob),
                A.Rotate(limit=CFG.rotation_degrees, p=0.5),
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


def get_df(phase: str, load_cached_data: bool = True):
    """
    Loads the dataframe for the specified phase, handling caching.

    Args:
        phase (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pandas.DataFrame: The loaded dataframe with full file paths.
    """
    cache_path = os.path.join(CFG.working_dir, f"{phase}_df.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to loading from source

    # Load from metadata
    if phase == "train":
        csv_path = CFG.train_csv
    elif phase == "val":
        csv_path = CFG.val_csv
    elif phase == "test":
        csv_path = CFG.test_csv
    else:
        raise ValueError(f"Unknown phase: {phase}")

    df = pd.read_csv(csv_path)

    # Construct full file paths
    # Metadata contains relative paths e.g., "train_images/id.png"
    # We need to prepend the input directory
    df["file_path"] = df["file_path"].apply(lambda x: os.path.join(CFG.input_dir, x))

    # Debug mode: reduce dataset size
    if CFG.debug:
        df = df.head(CFG.debug_sample_size).reset_index(drop=True)

    # Save to cache
    os.makedirs(CFG.working_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy detection.
    Handles image loading, circle cropping, and transformations.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (though verification script ensures they exist)
            # Create a black image of standard size to prevent crash
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Circle Crop
        image = circle_crop(image)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare output
        # For test set, we might not have diagnosis
        if "diagnosis" in row:
            label = torch.tensor(row["diagnosis"], dtype=torch.float32)
            return image, label
        else:
            return image
