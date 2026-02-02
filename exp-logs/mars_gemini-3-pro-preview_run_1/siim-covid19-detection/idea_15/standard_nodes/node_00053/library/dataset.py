import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ast

from library.config import Config
from library.utils import read_xray


def get_transforms(data_split):
    """
    Returns the Albumentations transformation pipeline for the given data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # Label-Consistent CoarseDropout:
                # mask_fill_value=0 ensures that if an opacity is dropped from the image,
                # it is also removed from the mask, preventing label noise.
                # Cite solution_lesson_node_00002, solution_lesson_node_00004
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    mask_fill_value=0,
                    p=Config.COARSE_DROPOUT_PROB,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ]
        )


def load_dataset_metadata(split, load_cached_data=True):
    """
    Loads and caches the metadata dataframe for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"metadata_{split}.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If cache load fails, fall back to source
            pass

    # 2. Load from source
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"Warning: Failed to cache metadata for {split}: {e}")

    return df


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-COVID19 Detection.
    Handles loading DICOM images, generating segmentation masks from bounding boxes,
    and applying consistent augmentations.
    """

    def __init__(self, split, transform=None, load_cached_data=True):
        self.split = split
        self.df = load_dataset_metadata(split, load_cached_data=load_cached_data)
        self.transform = transform or get_transforms(split)

        # Check if we have label columns (train/val sets)
        self.has_labels = "Negative for Pneumonia" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path is relative to input dir
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read original size image (grayscale, 0-255)
        # We pass size=None to get original dims so we can map boxes correctly
        img = read_xray(file_path, size=None)

        # Convert to RGB (H, W, 3) for ResNet backbone
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w = img.shape[:2]

        # 2. Create Mask
        # Initialize empty mask
        mask = np.zeros((h, w), dtype=np.float32)

        # If boxes exist, draw them on the mask
        if "boxes" in row and pd.notna(row["boxes"]):
            try:
                # boxes string format: [{'x':..., 'y':..., 'width':..., 'height':...}, ...]
                boxes = ast.literal_eval(row["boxes"])
                for box in boxes:
                    x = int(box["x"])
                    y = int(box["y"])
                    bw = int(box["width"])
                    bh = int(box["height"])

                    # Clip to image boundaries
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)

                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 1.0
            except Exception:
                # Fallback to empty mask on parse error
                pass

        # 3. Augmentations
        # Albumentations handles resizing of both image and mask
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask is tensor (1, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # 4. Labels
        if self.has_labels:
            labels = torch.tensor(
                [
                    row["Negative for Pneumonia"],
                    row["Typical Appearance"],
                    row["Indeterminate Appearance"],
                    row["Atypical Appearance"],
                ],
                dtype=torch.float32,
            )
        else:
            # Dummy labels for test set
            labels = torch.zeros(4, dtype=torch.float32)

        # Return dictionary for multi-task learning
        return {
            "image": img,
            "labels": labels,
            "mask": mask,
            "image_id": row["image_id"],
            "study_id": row["study_id"],
        }
