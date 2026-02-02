import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train' or 'val'/'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Conservative Geometric Augmentations
                # Rotation +/- 20 degrees, Scale 0.9-1.1
                # Shift limit small (0.05) to keep subject centered
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=(
                        Config.AUG_SCALE_MIN - 1.0,
                        Config.AUG_SCALE_MAX - 1.0,
                    ),
                    rotate_limit=Config.AUG_ROTATION,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.HorizontalFlip(p=0.5),
                # Photometric Augmentations
                # Brightness and Contrast only. Hue/Saturation excluded per strategy.
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS,
                    contrast_limit=Config.AUG_CONTRAST,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        # Resize and Normalize only.
        # TTA (Horizontal Flip) will be handled in the evaluation loop if needed.
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_class_list(load_cached_data=True):
    """
    Generates or loads the list of unique whale IDs (classes).
    Ensures deterministic label encoding across runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        list: Sorted list of unique class names.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load from cache
            classes_arr = np.load(cache_path, allow_pickle=False)
            return classes_arr.tolist()
        except Exception:
            # If load fails, fall through to recompute
            pass

    # Compute from scratch using training metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_CSV}")

    df = pd.read_csv(Config.TRAIN_CSV)
    classes = sorted(df["Id"].unique().tolist())

    # Save to cache
    np.save(cache_path, np.array(classes))

    return classes


class WhaleDataset(Dataset):
    def __init__(self, csv_file, mode="train", transform=None, class_list=None):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            class_list (list): List of unique class names for encoding. Required for train/val.
        """
        self.df = pd.read_csv(csv_file)
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Handle Debugging
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SUBSET_SIZE).copy()

        # Setup Label Encoding
        if self.mode in ["train", "val"]:
            if class_list is None:
                raise ValueError(
                    "class_list must be provided for training/validation mode."
                )
            # Create mapping from class name to integer index
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(class_list)
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains 'file_path' relative to input dir
        file_path = row["file_path"]
        full_path = os.path.join(self.input_dir, file_path)

        # Read Image
        image = cv2.imread(full_path)

        if image is None:
            # Fallback for missing images (should not happen given metadata checks)
            # Return a black image of correct size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode in ["train", "val"]:
            label_str = row["Id"]
            label = self.class_to_idx[label_str]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: return image and original filename (for submission)
            image_id = row["Image"]
            return image, image_id
