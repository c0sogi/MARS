import os
import ast
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import read_dicom


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    Handles images, segmentation masks, and study-level labels.
    """

    def __init__(self, images, masks, labels, transforms=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        if self.transforms:
            # Albumentations expects mask to be (H, W) or (H, W, C)
            # Our mask is (H, W), which is fine.
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W) for PyTorch
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[2] == 1:
            # If albumentations returns (H, W, 1), permute to (1, H, W)
            mask = mask.permute(2, 0, 1)

        # Label is already a numpy array, convert to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, mask, label


def get_transforms(data):
    """
    Returns Albumentations transforms for 'train' or 'val'.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.2
                ),
                # Label-Consistent CoarseDropout
                # mask_fill_value=0 ensures that if an opacity is occluded,
                # it is removed from the ground truth mask.
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "val":
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), A.Normalize(), ToTensorV2()]
        )


def process_data(
    metadata_path,
    cache_images_path,
    cache_masks_path,
    cache_labels_path,
    load_cached_data=True,
):
    """
    Processes DICOMs and Metadata into Numpy arrays.
    Implements caching to speed up subsequent runs.
    """
    # 1. Check Cache
    if load_cached_data:
        if (
            os.path.exists(cache_images_path)
            and os.path.exists(cache_masks_path)
            and os.path.exists(cache_labels_path)
        ):
            print(f"Loading cached data from {os.path.dirname(cache_images_path)}...")
            try:
                images = np.load(cache_images_path)
                masks = np.load(cache_masks_path)
                labels = np.load(cache_labels_path)
                print(f"Loaded {len(images)} samples.")
                return images, masks, labels
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data...")
        else:
            print("Cache not found. Processing data from scratch...")

    # 2. Process from Scratch
    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays
    num_samples = len(df)
    img_size = Config.IMG_SIZE

    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.uint8)
    masks = np.zeros((num_samples, img_size, img_size), dtype=np.float32)
    labels = np.zeros((num_samples, Config.NUM_CLASSES), dtype=np.float32)

    print(f"Processing {num_samples} images from {metadata_path}...")

    for i, row in df.iterrows():
        # A. Load Image
        # Read full size first to generate accurate masks
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img_full = read_dicom(full_path, size=None)  # Returns (H, W, 3)

        h_orig, w_orig = img_full.shape[:2]

        # B. Generate Mask
        mask_full = np.zeros((h_orig, w_orig), dtype=np.uint8)

        box_str = row["boxes"]
        if pd.notna(box_str):
            try:
                boxes = ast.literal_eval(box_str)
                for box in boxes:
                    # box format: {'x': ..., 'y': ..., 'width': ..., 'height': ...}
                    x = int(box["x"])
                    y = int(box["y"])
                    w = int(box["width"])
                    h = int(box["height"])

                    cv2.rectangle(mask_full, (x, y), (x + w, y + h), 1, -1)
            except:
                pass  # No boxes or parse error -> empty mask

        # C. Resize both
        img_resized = cv2.resize(
            img_full, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask_full, (img_size, img_size), interpolation=cv2.INTER_NEAREST
        )

        images[i] = img_resized
        masks[i] = mask_resized

        # D. Process Labels
        # Columns: Negative for Pneumonia, Typical Appearance, Indeterminate Appearance, Atypical Appearance
        # Config.CLASS_LABELS order matches the columns
        label_vec = row[Config.CLASS_LABELS].values.astype(np.float32)
        labels[i] = label_vec

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{num_samples}")

    # 3. Save to Cache
    print("Saving processed data to cache...")
    os.makedirs(os.path.dirname(cache_images_path), exist_ok=True)
    np.save(cache_images_path, images)
    np.save(cache_masks_path, masks)
    np.save(cache_labels_path, labels)
    print("Data processing complete.")

    return images, masks, labels


def get_dataloaders(load_cached_data=True):
    """
    Prepares Training and Validation DataLoaders.
    """
    # Process Train Data
    train_images, train_masks, train_labels = process_data(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMAGES,
        Config.CACHE_TRAIN_MASKS,
        Config.CACHE_TRAIN_LABELS,
        load_cached_data=load_cached_data,
    )

    # Process Val Data
    val_images, val_masks, val_labels = process_data(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_IMAGES,
        Config.CACHE_VAL_MASKS,
        Config.CACHE_VAL_LABELS,
        load_cached_data=load_cached_data,
    )

    # Create Datasets
    train_dataset = SIIMDataset(
        train_images, train_masks, train_labels, transforms=get_transforms("train")
    )

    val_dataset = SIIMDataset(
        val_images, val_masks, val_labels, transforms=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
