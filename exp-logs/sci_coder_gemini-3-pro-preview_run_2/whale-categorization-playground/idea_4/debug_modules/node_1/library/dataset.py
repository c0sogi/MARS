import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Whale Species Prediction.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray, optional): Array of integer labels (N,).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, torch.tensor(-1, dtype=torch.long)  # Dummy label for test


def get_transforms(mode="train"):
    """
    Returns the Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_and_preprocess_images(df, cache_path, load_cached_data=True):
    """
    Loads images from disk, resizes them, and caches the result.

    Args:
        df (pd.DataFrame): DataFrame containing 'file_path'.
        cache_path (str): Path to save/load the .npy cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of processed images (N, H, W, 3) uint8.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached images from {cache_path}...")
            images = np.load(cache_path)
            if len(images) == len(df):
                return images
            else:
                print(
                    f"Cache size mismatch ({len(images)} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {len(df)} images for {cache_path}...")
    images = []
    missing_count = 0

    for idx, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is None:
                # Create a black image if corrupt
                img = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
                )
                missing_count += 1
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        else:
            # Create a black image if missing
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
            missing_count += 1

        images.append(img)

    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing or corrupt.")

    images = np.array(images, dtype=np.uint8)

    # 3. Save to cache
    print(f"Saving cache to {cache_path}...")
    np.save(cache_path, images)

    return images


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader, label_encoder, num_classes
    """
    print("Initializing Data Loaders...")

    # ---------------------------------------------------------
    # 1. Load Metadata
    # ---------------------------------------------------------
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # ---------------------------------------------------------
    # 2. Filter Training Data (Exclude new_whale)
    # ---------------------------------------------------------
    # As per requirements: "The training set will strictly exclude the new_whale class."
    train_df = train_df[train_df["Id"] != "new_whale"].reset_index(drop=True)
    print(f"Filtered Training Samples (Known Whales Only): {len(train_df)}")

    # ---------------------------------------------------------
    # 3. Debug Mode
    # ---------------------------------------------------------
    if debug:
        print(f"DEBUG MODE: Limiting to {Config.DEBUG_SAMPLES} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        test_df = test_df.head(Config.DEBUG_SAMPLES)

        # Adjust cache paths for debug to avoid overwriting full cache
        train_cache = Config.CACHE_TRAIN_IMAGES.replace(".npy", "_debug.npy")
        val_cache = Config.CACHE_VAL_IMAGES.replace(".npy", "_debug.npy")
        test_cache = Config.CACHE_TEST_IMAGES.replace(".npy", "_debug.npy")
    else:
        train_cache = Config.CACHE_TRAIN_IMAGES
        val_cache = Config.CACHE_VAL_IMAGES
        test_cache = Config.CACHE_TEST_IMAGES

    # ---------------------------------------------------------
    # 4. Label Encoding
    # ---------------------------------------------------------
    # Create mapping from Id to Int based on Training Data
    unique_ids = sorted(train_df["Id"].unique())
    label_encoder = {label: idx for idx, label in enumerate(unique_ids)}
    num_classes = len(unique_ids)
    print(f"Number of Classes (Known Whales): {num_classes}")

    # Encode Labels
    # For training, all should map successfully
    train_labels = train_df["Id"].map(label_encoder).values.astype(np.int64)

    # For validation, 'new_whale' or unseen classes map to -1
    val_labels = (
        val_df["Id"].map(lambda x: label_encoder.get(x, -1)).values.astype(np.int64)
    )

    # ---------------------------------------------------------
    # 5. Load Images (with Caching)
    # ---------------------------------------------------------
    train_images = load_and_preprocess_images(train_df, train_cache, load_cached_data)
    val_images = load_and_preprocess_images(val_df, val_cache, load_cached_data)
    test_images = load_and_preprocess_images(test_df, test_cache, load_cached_data)

    # ---------------------------------------------------------
    # 6. Create Datasets
    # ---------------------------------------------------------
    train_dataset = WhaleDataset(
        images=train_images, labels=train_labels, transform=get_transforms("train")
    )

    val_dataset = WhaleDataset(
        images=val_images, labels=val_labels, transform=get_transforms("val")
    )

    test_dataset = WhaleDataset(
        images=test_images,
        labels=None,  # No labels for test
        transform=get_transforms("test"),
    )

    # ---------------------------------------------------------
    # 7. Create Loaders
    # ---------------------------------------------------------
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, label_encoder, num_classes
