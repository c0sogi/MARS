import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads a mapping from class names (whale Ids) to integer indices.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (class_to_idx (dict), classes (list))
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path, allow_pickle=True)
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            return class_to_idx, classes.tolist()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_CSV}")

    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Get unique IDs and sort them for determinism
    classes = sorted(df_train["Id"].unique().tolist())

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, np.array(classes))

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    return class_to_idx, classes


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_size = Config.IMG_SIZE

    if phase == "train":
        # Calculate scale limit for ShiftScaleRotate (e.g., 0.9-1.1 -> limit 0.1)
        # Config: AUG_SCALE_MIN = 0.9, AUG_SCALE_MAX = 1.1
        # Albumentations scale_limit is +/- factor.
        # We use the larger deviation to define the limit.
        scale_limit = max(
            abs(1.0 - Config.AUG_SCALE_MIN), abs(Config.AUG_SCALE_MAX - 1.0)
        )

        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                # Geometric Augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.0,  # No shifting, just scale/rotate
                    scale_limit=scale_limit,
                    rotate_limit=Config.AUG_ROTATION,
                    # Reduced probability to preserve input identity (Cite solution_lesson_node_00030)
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.HorizontalFlip(p=Config.AUG_HFLIP_PROB),
                # Photometric Augmentations (Brightness/Contrast only)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization and Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    def __init__(
        self, mode, transform=None, debug=Config.DEBUG, load_cached_mapping=True
    ):
        """
        Custom Dataset for Whale Identification.

        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transform pipeline.
            debug (bool): If True, limits dataset size for debugging.
            load_cached_mapping (bool): Whether to use cached class mapping.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Select Metadata File
        if mode == "train":
            self.csv_path = Config.TRAIN_CSV
        elif mode == "val":
            self.csv_path = Config.VAL_CSV
        elif mode == "test":
            self.csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load Metadata
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # Handle Debugging
        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLES].copy()

        # Handle Labels for Train/Val
        self.class_to_idx = None
        if mode in ["train", "val"]:
            self.class_to_idx, _ = get_class_mapping(
                load_cached_data=load_cached_mapping
            )

            # Validate that all IDs in current df exist in the mapping
            # (Validation set is a subset of Train, so this should pass)
            # We map the IDs immediately to avoid lookups in __getitem__?
            # No, keep dynamic to handle potential data issues gracefully or debug.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains 'file_path' relative to input dir
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Read Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though verification script checks this)
            # Return a blank image to prevent crash, or raise error
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in pipeline)
            transform = ToTensorV2()
            image_tensor = transform(image=image)["image"]

        # Return based on mode
        if self.mode == "test":
            # Return image and filename (ID) for submission
            return image_tensor, row["Image"]
        else:
            # Return image and integer label
            label_name = row["Id"]
            label_idx = self.class_to_idx[label_name]
            return image_tensor, torch.tensor(label_idx, dtype=torch.long)
