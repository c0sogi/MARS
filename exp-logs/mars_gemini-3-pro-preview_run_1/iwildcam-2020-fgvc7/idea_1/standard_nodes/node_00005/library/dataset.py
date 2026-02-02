import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MEGADETECTOR_PATH,
    IMAGE_SIZE,
)
from library.utils import load_megadetector_data


def get_transforms(image_size, mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
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


class CameraTrapDataset(Dataset):
    def __init__(self, df, bbox_map, transform=None, root_dir=INPUT_DIR, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            bbox_map (dict): Dictionary mapping image_id to [x, y, w, h].
            transform (albumentations.Compose): Transforms to apply.
            root_dir (str): Root directory for images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.bbox_map = bbox_map
        self.transform = transform
        self.root_dir = root_dir
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _crop_roi(self, img, bbox):
        """
        Crops the image based on the bounding box with a small margin.
        bbox is [x, y, w, h] in normalized coordinates (0-1).
        """
        h_img, w_img = img.shape[:2]
        x, y, w, h = bbox

        # Add 5% context margin
        margin = 0.05
        pad_w = w * margin
        pad_h = h * margin

        # Calculate coordinates with clamping
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(1.0, x + w + pad_w)
        y2 = min(1.0, y + h + pad_h)

        # Convert to pixels
        x1_px = int(x1 * w_img)
        y1_px = int(y1 * h_img)
        x2_px = int(x2 * w_img)
        y2_px = int(y2 * h_img)

        # Sanity check: if crop is invalid or empty, return full image
        if x2_px <= x1_px or y2_px <= y1_px:
            return img

        return img[y1_px:y2_px, x1_px:x2_px]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        full_path = os.path.join(self.root_dir, file_path)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for corrupt/missing images: black image
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get bounding box (default to full image if missing)
        bbox = self.bbox_map.get(image_id, [0.0, 0.0, 1.0, 1.0])

        # Crop to RoI
        img = self._crop_roi(img, bbox)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Return logic based on mode
        if self.mode == "test":
            # For test, we need the image_id for submission
            return img, image_id
        else:
            # For train/val, we need the category label
            label = row["category_id"]
            return img, torch.tensor(label, dtype=torch.long)


def create_datasets(load_cached_data=True):
    """
    Loads metadata and MegaDetector results, then creates Train, Val, and Test datasets.

    Args:
        load_cached_data (bool): Whether to load cached MegaDetector results.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 1. Load Metadata
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # 2. Load MegaDetector Bounding Boxes
    bbox_map = load_megadetector_data(
        MEGADETECTOR_PATH, load_cached_data=load_cached_data
    )

    # 3. Create Transforms
    train_transform = get_transforms(IMAGE_SIZE, mode="train")
    eval_transform = get_transforms(IMAGE_SIZE, mode="val")

    # 4. Instantiate Datasets
    print("Creating datasets...")
    train_dataset = CameraTrapDataset(
        train_df, bbox_map, transform=train_transform, mode="train"
    )

    val_dataset = CameraTrapDataset(
        val_df, bbox_map, transform=eval_transform, mode="val"
    )

    test_dataset = CameraTrapDataset(
        test_df, bbox_map, transform=eval_transform, mode="test"
    )

    return train_dataset, val_dataset, test_dataset
