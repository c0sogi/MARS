import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformations for the specified split.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                # Cutout or CoarseDropout can be effective for occlusion invariance
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, label_encoder=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C) or (N, H, W)
            labels (list/pd.Series): List of string IDs.
            transform (A.Compose): Albumentations transforms.
            label_encoder (LabelEncoder): Fitted encoder to convert string IDs to ints.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Labels
        if self.labels is not None:
            label_str = self.labels[idx]

            # Encode label
            if self.label_encoder:
                try:
                    # Transform returns an array, we take the first item
                    target = self.label_encoder.transform([label_str])[0]
                except ValueError:
                    # Handle unseen labels (e.g., 'new_whale' in validation)
                    target = -1
            else:
                # If no encoder provided, return string (or whatever is in labels)
                target = label_str

            return image, target
        else:
            return image


def load_data_and_cache(df, cache_path, input_dir, load_cached=True):
    """
    Loads images based on the dataframe paths, resizes them, and caches to disk.
    If cache exists and load_cached is True, loads from disk.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Simple integrity check
            if len(data) == len(df):
                print(f"Loaded {len(data)} images from cache: {cache_path}")
                return data
            else:
                print(
                    f"Cache mismatch (Size {len(data)} vs DF {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {len(df)} images for {os.path.basename(cache_path)}...")
    images = []

    # Pre-allocate if possible, but list append is usually fine for this scale
    # Processing loop
    for _, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input_dir (e.g., "train/xxxx.jpg")
        full_path = os.path.join(input_dir, row["file_path"])

        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images (should ideally not happen given EDA)
            # Create a black image
            img = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (Config.image_size, Config.image_size))

        images.append(img)

    data = np.array(images, dtype=np.uint8)

    # 3. Save to cache
    try:
        np.save(cache_path, data)
        print(f"Saved cache to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return data


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation (Query), Gallery, and Test.

    Returns:
        train_loader, val_loader, gallery_loader, test_loader, num_classes, label_encoder
    """
    print("Initializing Data Loaders...")

    # ---------------------------------------------------------
    # 1. Load Metadata
    # ---------------------------------------------------------
    train_df = pd.read_csv(Config.train_csv_path)
    val_df = pd.read_csv(Config.val_csv_path)
    test_df = pd.read_csv(Config.test_csv_path)

    # ---------------------------------------------------------
    # 2. Filter Training Data
    # ---------------------------------------------------------
    # Exclude 'new_whale' from training set
    train_df_filtered = train_df[train_df["Id"] != "new_whale"].reset_index(drop=True)
    print(f"Filtered Training Samples (Known Whales Only): {len(train_df_filtered)}")

    # ---------------------------------------------------------
    # 3. Label Encoding
    # ---------------------------------------------------------
    # Fit encoder ONLY on known whales
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df_filtered["Id"])
    num_classes = len(label_encoder.classes_)
    print(f"Number of Known Classes: {num_classes}")

    # ---------------------------------------------------------
    # 4. Image Loading & Caching
    # ---------------------------------------------------------
    # Note: We use specific cache names for the filtered training set

    # Train Images (Filtered)
    train_cache_path = Config.train_images_cache
    train_images = load_data_and_cache(
        train_df_filtered,
        train_cache_path,
        Config.input_dir,
        load_cached=load_cached_data,
    )

    # Val Images
    val_cache_path = Config.val_images_cache
    val_images = load_data_and_cache(
        val_df, val_cache_path, Config.input_dir, load_cached=load_cached_data
    )

    # Test Images
    test_cache_path = Config.test_images_cache
    test_images = load_data_and_cache(
        test_df, test_cache_path, Config.input_dir, load_cached=load_cached_data
    )

    # ---------------------------------------------------------
    # 5. Dataset Creation
    # ---------------------------------------------------------

    # Train Dataset: Augmented, Known Whales
    train_dataset = WhaleDataset(
        images=train_images,
        labels=train_df_filtered["Id"].values,
        transform=get_transforms("train"),
        label_encoder=label_encoder,
    )

    # Gallery Dataset: Deterministic, Known Whales (Same as Train data but no aug)
    # Used for retrieval database
    gallery_dataset = WhaleDataset(
        images=train_images,
        labels=train_df_filtered["Id"].values,
        transform=get_transforms("val"),  # Deterministic
        label_encoder=label_encoder,
    )

    # Val Dataset: Deterministic, Includes new_whale
    # Used as Queries
    val_dataset = WhaleDataset(
        images=val_images,
        labels=val_df["Id"].values,
        transform=get_transforms("val"),
        label_encoder=label_encoder,
    )

    # Test Dataset: Deterministic, No Labels
    test_dataset = WhaleDataset(
        images=test_images,
        labels=None,  # No labels for test
        transform=get_transforms("val"),
        label_encoder=None,
    )

    # ---------------------------------------------------------
    # 6. DataLoader Creation
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for BatchNorm stability
    )

    # Gallery loader (larger batch size possible as no backprop)
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        gallery_loader,
        test_loader,
        num_classes,
        label_encoder,
    )
