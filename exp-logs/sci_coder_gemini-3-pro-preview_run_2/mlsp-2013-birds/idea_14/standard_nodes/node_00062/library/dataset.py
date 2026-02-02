import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_transforms(resize_dims, data_type="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        resize_dims (tuple): (height, width) for resizing.
        data_type (str): 'train' for augmentation, 'val' or 'test' for deterministic.
    """
    height, width = resize_dims

    # Normalization statistics (ImageNet defaults work well for pseudo-RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_type == "train":
        return A.Compose(
            [
                A.Resize(height, width),
                # SpecAugment-like CoarseDropout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(height * 0.1),
                    max_width=int(width * 0.1),
                    min_holes=None,
                    fill_value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def load_images_with_cache(df, cache_name, load_cached_data=True):
    """
    Loads images from disk or cache.

    Args:
        df (pd.DataFrame): DataFrame containing 'file_path_spec'.
        cache_name (str): Unique identifier for the cache file (e.g., 'train_images').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of images with shape (N, H, W).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading images from cache: {cache_path}")
            images = np.load(cache_path)
            if len(images) == len(df):
                return images
            else:
                pass  # Cache mismatch, reload
        except Exception:
            pass  # Corrupt cache, reload

    # 2. Load from scratch
    # print(f"Loading images from disk for {cache_name}...")
    images = []

    # We need to construct the path to the filtered spectrograms
    # The metadata points to 'supplemental_data/spectrograms/...'
    # We want 'supplemental_data/filtered_spectrograms/...'
    # Or simply use the filename and join with Config.FILTERED_SPEC_DIR

    for idx, row in df.iterrows():
        rel_path = row["file_path_spec"]
        filename = os.path.basename(rel_path)
        full_path = os.path.join(Config.FILTERED_SPEC_DIR, filename)

        if not os.path.exists(full_path):
            # Fallback to original path if filtered not found (should not happen based on task)
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Placeholder for missing files (should crash/error ideally, but handling gracefully)
            # Creating a black image of average size found in EDA (256x1246)
            img = np.zeros((256, 1246), dtype=np.uint8)

        images.append(img)

    # Stack into numpy array. Note: Images might have different widths.
    # If widths differ, we cannot stack into a perfect (N, H, W) array without resizing first.
    # Based on EDA, widths are constant? EDA said "Width Mean: 1246.0000, Std: 0.0000".
    # So we can stack.
    try:
        images = np.array(images)
    except ValueError:
        # If dimensions mismatch, keep as object array (list of arrays)
        # But we can't save object array easily to npy without allow_pickle=True which is discouraged
        # However, EDA suggests constant dimensions.
        images = np.array(images, dtype=object)

    # 3. Save to cache
    np.save(cache_path, images)

    return images


class BirdDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, soft_labels=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            labels (np.ndarray, optional): Hard labels (N, NumClasses).
            transforms (A.Compose, optional): Albumentations transforms.
            soft_labels (np.ndarray, optional): Soft labels for semi-supervised learning (N, NumClasses).
                                                If provided, these take precedence over 'labels'.
        """
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get image
        img = self.images[idx]

        # 2. Pseudo-RGB Conversion (Channel Replication)
        # Input is (H, W), Output needed is (H, W, 3) for Albumentations/Models
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        # 3. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Fallback to tensor conversion
            img = ToTensorV2()(image=img)["image"]

        # 4. Get Target
        target = None
        if self.soft_labels is not None:
            target = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        elif self.labels is not None:
            target = torch.tensor(self.labels[idx], dtype=torch.float32)

        if target is not None:
            return img, target
        else:
            return img, torch.zeros(
                Config.NUM_CLASSES
            )  # Dummy target for test set if no labels


def get_dataloaders(
    model_name,
    train_df=None,
    val_df=None,
    test_df=None,
    pseudo_labels=None,
    load_cached_data=True,
):
    """
    Creates DataLoaders for the specified data splits.

    Args:
        model_name (str): Name of the model to determine resolution.
        train_df (pd.DataFrame, optional): Training metadata.
        val_df (pd.DataFrame, optional): Validation metadata.
        test_df (pd.DataFrame, optional): Test metadata.
        pseudo_labels (np.ndarray, optional): Soft labels for the test set (for semi-supervised training).
                                              If provided, test_df is treated as additional training data.
        load_cached_data (bool): Whether to use cached image arrays.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders if corresponding DFs are provided.
    """
    dataloaders = {}
    resolution = Config.get_resolution(model_name)

    # --- Training Loader ---
    if train_df is not None:
        # Load Train Images
        train_imgs = load_images_with_cache(
            train_df, cache_name="train_images", load_cached_data=load_cached_data
        )

        # Extract Hard Labels
        label_cols = [c for c in train_df.columns if c.startswith("species_")]
        train_labels = train_df[label_cols].values.astype(np.float32)

        # If pseudo_labels are provided, we are in Semi-Supervised Mode (Student Training)
        # We need to combine train_df (hard labels) and test_df (soft labels)
        if pseudo_labels is not None and test_df is not None:
            # Load Test Images (to act as extra training data)
            test_imgs = load_images_with_cache(
                test_df, cache_name="test_images", load_cached_data=load_cached_data
            )

            # Combine Images
            combined_imgs = np.concatenate([train_imgs, test_imgs], axis=0)

            # Combine Labels
            # For original train data: Hard labels -> Soft format (0.0 or 1.0)
            # For pseudo test data: Soft labels
            combined_labels = np.concatenate([train_labels, pseudo_labels], axis=0)

            train_dataset = BirdDataset(
                images=combined_imgs,
                soft_labels=combined_labels,  # Use soft_labels argument for unified handling
                transforms=get_transforms(resolution, data_type="train"),
            )
        else:
            # Standard Supervised Training
            train_dataset = BirdDataset(
                images=train_imgs,
                labels=train_labels,
                transforms=get_transforms(resolution, data_type="train"),
            )

        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=lambda worker_id: seed_everything(Config.SEED + worker_id),
        )

    # --- Validation Loader ---
    if val_df is not None:
        val_imgs = load_images_with_cache(
            val_df, cache_name="val_images", load_cached_data=load_cached_data
        )
        label_cols = [c for c in val_df.columns if c.startswith("species_")]
        val_labels = val_df[label_cols].values.astype(np.float32)

        val_dataset = BirdDataset(
            images=val_imgs,
            labels=val_labels,
            transforms=get_transforms(resolution, data_type="val"),
        )

        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    # --- Test Loader ---
    if test_df is not None and pseudo_labels is None:
        # Only create a separate test loader if we aren't using test_df for training
        test_imgs = load_images_with_cache(
            test_df, cache_name="test_images", load_cached_data=load_cached_data
        )
        # Labels are placeholders in test_df, but dataset expects them or handles None
        # We pass None for labels to be safe, or the zeros from DF

        test_dataset = BirdDataset(
            images=test_imgs,
            labels=None,  # No ground truth
            transforms=get_transforms(resolution, data_type="test"),
        )

        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return dataloaders
