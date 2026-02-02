import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library import config
from library import box_utils


def get_transforms(split):
    """
    Returns the Albumentations transform pipeline based on the data split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                # Resize is handled after cropping, but we ensure input to model is fixed size
                A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
                # Aggressive Augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                # Regularization
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(config.IMG_SIZE * 0.1),
                    max_width=int(config.IMG_SIZE * 0.1),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CameraTrapDataset(Dataset):
    def __init__(self, split, transform=None, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transform pipeline.
            load_cached_data (bool): Whether to load cached MegaDetector data.
        """
        self.split = split
        self.transform = transform

        # 1. Load Metadata
        if split == "train":
            self.metadata = pd.read_csv(config.TRAIN_METADATA_PATH)
        elif split == "val":
            self.metadata = pd.read_csv(config.VAL_METADATA_PATH)
        elif split == "test":
            self.metadata = pd.read_csv(config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Invalid split: {split}")

        # 2. Load MegaDetector Data
        self.bbox_df = box_utils.load_megadetector_data(
            load_cached_data=load_cached_data
        )

        # 3. Merge Metadata with BBox Data
        # We merge on image_id. Left join ensures we keep all images in the split.
        self.data = pd.merge(self.metadata, self.bbox_df, on="image_id", how="left")

        # Fill missing bboxes (if any image wasn't in megadetector results) with NaNs
        # box_utils.load_megadetector_data already handles missing detections by returning NaNs/0 conf
        # but the merge might introduce NaNs if an ID is missing entirely from bbox_df.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image_id"]
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        # 1. Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images: create black image
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape

        # 2. Get Bounding Box Info
        # Check confidence threshold
        conf = row.get("conf", 0.0)
        if pd.isna(conf):
            conf = 0.0

        bbox = None
        if conf >= config.DETECTION_CONF_THRESHOLD:
            bbox_vals = [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]
            if not any(np.isnan(bbox_vals)):
                bbox = bbox_vals

        # 3. Calculate Crop Coordinates
        # returns x_min, y_min, x_max, y_max (absolute pixels)
        x_min, y_min, x_max, y_max = box_utils.get_context_square_crop(
            bbox, w, h, margin=config.CROP_MARGIN
        )

        # 4. Pad and Crop
        # Calculate required padding if crop extends beyond image
        pad_left = max(0, -x_min)
        pad_top = max(0, -y_min)
        pad_right = max(0, x_max - w)
        pad_bottom = max(0, y_max - h)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            image = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        # Adjust crop coordinates to the padded image
        crop_x = x_min + pad_left
        crop_y = y_min + pad_top
        crop_w = x_max - x_min
        crop_h = y_max - y_min

        # Perform crop
        image = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        # Safety check: if crop resulted in empty image (shouldn't happen with logic above)
        if image.size == 0:
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)

        # 5. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 6. Return Data
        if self.split in ["train", "val"]:
            label = row["category_id"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test, we might need image_id for submission construction downstream,
            # but standard PyTorch datasets usually return tensors.
            # The caller (inference loop) usually iterates the dataset sequentially.
            return image, image_id


def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Datasets
    train_dataset = CameraTrapDataset("train", transform=train_transform)
    val_dataset = CameraTrapDataset("val", transform=val_transform)
    test_dataset = CameraTrapDataset("test", transform=val_transform)

    # Loaders
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
