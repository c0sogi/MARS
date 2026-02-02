import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from timm.data.mixup import Mixup

from library.config import Config
from library.utils import load_megadetector_data


class IWildCamDataset(Dataset):
    """
    PyTorch Dataset for iWildCam 2020.
    Handles loading images, cropping based on MegaDetector boxes,
    resizing, and applying augmentations.
    """

    def __init__(self, df, bbox_dict, root_dir, transform=None, is_test=False):
        self.df = df
        self.bbox_dict = bbox_dict
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        # Pre-extract lists for faster access
        self.ids = self.df["id"].values
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            self.labels = self.df["category_id"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        file_path = self.file_paths[idx]

        # Construct full image path
        full_path = os.path.join(self.root_dir, file_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing/corrupt images: create black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get Bounding Box (normalized [x, y, w, h])
        # Default to full image if ID not found
        bbox = self.bbox_dict.get(img_id, [0.0, 0.0, 1.0, 1.0])

        # Crop Image
        h_img, w_img, _ = image.shape
        x_norm, y_norm, w_norm, h_norm = bbox

        x_min = int(x_norm * w_img)
        y_min = int(y_norm * h_img)
        w_crop = int(w_norm * w_img)
        h_crop = int(h_norm * h_img)

        # Ensure coordinates are within bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        # Perform crop if valid
        if w_crop > 0 and h_crop > 0:
            crop = image[y_min : y_min + h_crop, x_min : x_min + w_crop]
            # Check if crop is empty (can happen with rounding)
            if crop.size == 0:
                crop = image
        else:
            crop = image

        image = crop

        # Apply Transforms (Resize, Augment, Normalize, ToTensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data
        if self.is_test:
            # Return dummy label for test
            return image, torch.tensor(0, dtype=torch.long)
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size, mode="train"):
    """
    Returns Albumentations transforms for train or validation/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                # Conservative color jitter to prevent overfitting to specific lighting
                # but preserving texture details
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                ),
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
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Also initializes and returns the Mixup function for training.

    Args:
        debug (bool): If True, subsamples datasets for quick testing.

    Returns:
        train_loader, val_loader, test_loader, mixup_fn
    """
    print(f"Loading metadata from {Config.METADATA_DIR}...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Load MegaDetector Boxes (Cached)
    print("Loading MegaDetector bounding boxes...")
    bbox_dict = load_megadetector_data(
        json_path=Config.MEGADETECTOR_FILE,
        cache_dir=Config.WORKING_DIR,
        load_cached_data=True,
    )

    # Debug Subsampling
    if debug:
        print(
            f"DEBUG MODE: Subsampling datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # Test set is usually needed in full for submission, but for debug flow we can subsample
        # However, usually we want to verify the full pipeline even in debug for test output format
        # We'll subsample test as well to save time if debug is on
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define Transforms
    train_transform = get_transforms(Config.IMG_SIZE, mode="train")
    val_transform = get_transforms(Config.IMG_SIZE, mode="val")

    # Instantiate Datasets
    train_dataset = IWildCamDataset(
        df_train, bbox_dict, Config.INPUT_DIR, transform=train_transform, is_test=False
    )
    val_dataset = IWildCamDataset(
        df_val, bbox_dict, Config.INPUT_DIR, transform=val_transform, is_test=False
    )
    test_dataset = IWildCamDataset(
        df_test, bbox_dict, Config.INPUT_DIR, transform=val_transform, is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm/Mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Mixup
    # Mixup is applied within the training loop on batches
    mixup_fn = Mixup(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=Config.LABEL_SMOOTHING,
        num_classes=Config.NUM_CLASSES,
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, mixup_fn
