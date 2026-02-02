import os
import random
import numpy as np
import torch
import pydicom
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dicom_slice(path: str) -> np.ndarray:
    """
    Loads a DICOM file, converts it to Hounsfield Units (HU), applies a bone-specific
    windowing function, and normalizes the result to the [0, 1] range.

    Args:
        path (str): The file path to the DICOM image.

    Returns:
        np.ndarray: A float32 numpy array of shape (H, W) with values in [0, 1].
                    Returns a black image if loading fails.
    """
    try:
        if not os.path.exists(path):
            # Return blank image if file is missing
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

        ds = pydicom.dcmread(path)

        # Extract pixel array
        if not hasattr(ds, "pixel_array"):
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

        img = ds.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU) using RescaleSlope and RescaleIntercept
        intercept = getattr(ds, "RescaleIntercept", 0)
        slope = getattr(ds, "RescaleSlope", 1)
        img = img * slope + intercept

        # Apply Bone Windowing
        # Standard Bone Window: Center (WL) = 500, Width (WW) = 2000
        # This maps the range [-500, 1500] to visible contrast.
        window_center = 500
        window_width = 2000

        min_value = window_center - (window_width / 2)
        max_value = window_center + (window_width / 2)

        img = np.clip(img, min_value, max_value)

        # Normalize to [0, 1]
        img = (img - min_value) / (max_value - min_value)

        return img.astype(np.float32)

    except Exception as e:
        print(f"Error loading DICOM {path}: {e}")
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)


def get_transforms(data: str = "train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): The data split ('train', 'valid', or 'test').

    Returns:
        A.ReplayCompose (for train) or A.Compose: The transformation pipeline.
        ReplayCompose is used for training to allow the Dataset to apply the exact same
        geometric augmentation parameters across the entire sequence of 2.5D stacks
        (volumetric consistency).
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data == "train":
        return A.ReplayCompose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

    elif data in ["val", "valid", "test"]:
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(f"Unknown data split: {data}")
