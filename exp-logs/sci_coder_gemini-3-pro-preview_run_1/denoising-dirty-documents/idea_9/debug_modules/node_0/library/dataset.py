import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_images(metadata_path, cache_name, load_cached_data=True):
    """
    Loads images based on metadata. Implements caching using .npz files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_name (str): Name of the cache file (e.g., 'train_cache').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (data_df, images_dict)
            - data_df: DataFrame containing metadata.
            - images_dict: Dictionary mapping relative path -> numpy image array (grayscale).
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading data from cache: {cache_path}")
            loaded = np.load(cache_path, allow_pickle=True)
            # Reconstruct dictionary from arrays
            # We stored keys and values as separate arrays in the npz
            keys = loaded["keys"]
            # values are stored as a flat object array of arrays because shapes vary
            values = loaded["values"]
            images_dict = {k: v for k, v in zip(keys, values)}

            # Load dataframe
            df_path = metadata_path
            if os.path.exists(df_path):
                data_df = pd.read_csv(df_path)
                return data_df, images_dict
            else:
                print("Metadata file missing despite cache existing. Re-generating.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-generating data.")

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    data_df = pd.read_csv(metadata_path)
    images_dict = {}

    # Collect all unique paths needed
    paths_to_load = []
    if "noisy_image_path" in data_df.columns:
        paths_to_load.extend(data_df["noisy_image_path"].tolist())
    if "clean_image_path" in data_df.columns:
        paths_to_load.extend(data_df["clean_image_path"].tolist())

    paths_to_load = list(set(paths_to_load))

    print(f"Loading {len(paths_to_load)} images from disk...")

    for rel_path in paths_to_load:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        # Load as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")
        images_dict[rel_path] = img

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    keys = np.array(list(images_dict.keys()))
    # Use object array for values since images have different shapes
    values = np.array(list(images_dict.values()), dtype=object)

    np.savez_compressed(cache_path, keys=keys, values=values)
    print(f"Saved data to cache: {cache_path}")

    return data_df, images_dict


class DenoisingDataset(Dataset):
    def __init__(
        self, metadata_df, images_dict, patch_size, mode="train", augment=True
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame with image paths.
            images_dict (dict): Dictionary mapping paths to image arrays.
            patch_size (int): Size of the crop (H=W=patch_size).
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply geometric augmentations (only for train).
        """
        self.metadata = metadata_df
        self.images_dict = images_dict
        self.patch_size = patch_size
        self.mode = mode
        self.augment = augment

        # Define Augmentations
        if self.mode == "train" and self.augment:
            self.transform = A.Compose(
                [
                    A.RandomCrop(
                        height=self.patch_size, width=self.patch_size, always_apply=True
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ]
            )
        elif self.mode == "val":
            # Deterministic Center Crop for validation to ensure batching
            self.transform = A.Compose(
                [
                    A.CenterCrop(
                        height=self.patch_size, width=self.patch_size, always_apply=True
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            # Test mode or inference: usually full image or specific handling
            # If patch_size is provided, we center crop. If None, we return full image.
            if self.patch_size:
                self.transform = A.Compose(
                    [
                        A.CenterCrop(
                            height=self.patch_size,
                            width=self.patch_size,
                            always_apply=True,
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load Noisy Image
        noisy_path = row["noisy_image_path"]
        noisy_img = self.images_dict[noisy_path]

        # Normalize to [0, 1] float32
        noisy_img = noisy_img.astype(np.float32) / 255.0

        if self.mode in ["train", "val"]:
            # Load Clean Image
            clean_path = row["clean_image_path"]
            clean_img = self.images_dict[clean_path]
            clean_img = clean_img.astype(np.float32) / 255.0

            # Apply transforms (Augmentation + Crop)
            # Albumentations expects HWC, but we have HW. Expand dims?
            # Actually Albumentations works fine with HW for grayscale if we don't force channel checks incorrectly.
            # However, ToTensorV2 usually expects HWC or HW.
            # Let's keep them as HW numpy arrays, pass to transform.

            augmented = self.transform(image=noisy_img, mask=clean_img)
            noisy_tensor = augmented["image"]
            clean_tensor = augmented["mask"]

            # Ensure channel dimension (1, H, W)
            if noisy_tensor.ndim == 2:
                noisy_tensor = noisy_tensor.unsqueeze(0)
            if clean_tensor.ndim == 2:
                clean_tensor = clean_tensor.unsqueeze(0)

            # Albumentations ToTensorV2 might not add channel dim for grayscale automatically if input was HW
            # It converts to tensor. If input was HW, output is HW. We need 1HW.
            if noisy_tensor.shape[0] != 1:
                noisy_tensor = noisy_tensor.unsqueeze(0)
            if clean_tensor.shape[0] != 1:
                clean_tensor = clean_tensor.unsqueeze(0)

            return noisy_tensor, clean_tensor, row["id"]

        else:
            # Test Mode (Noisy only)
            augmented = self.transform(image=noisy_img)
            noisy_tensor = augmented["image"]

            if noisy_tensor.ndim == 2:
                noisy_tensor = noisy_tensor.unsqueeze(0)
            if noisy_tensor.shape[0] != 1:
                noisy_tensor = noisy_tensor.unsqueeze(0)

            return noisy_tensor, row["id"]


def get_dataloaders(stream_config, load_cached_data=True, debug_limit=None):
    """
    Creates DataLoaders for a specific stream configuration.

    Args:
        stream_config (dict): Configuration dictionary for the stream (contains patch_size, etc.).
        load_cached_data (bool): Whether to use cached data loading.
        debug_limit (int, optional): Limit number of samples for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    patch_size = stream_config["patch_size"]
    batch_size = Config.BATCH_SIZE

    # 1. Load Data
    train_df, train_imgs = load_images(
        Config.TRAIN_METADATA, "train_cache", load_cached_data
    )
    val_df, val_imgs = load_images(Config.VAL_METADATA, "val_cache", load_cached_data)

    # Debugging limit
    if debug_limit or Config.MAX_TRAIN_SAMPLES:
        limit = debug_limit if debug_limit else Config.MAX_TRAIN_SAMPLES
        train_df = train_df.head(limit)

    if debug_limit or Config.MAX_VAL_SAMPLES:
        limit = debug_limit if debug_limit else Config.MAX_VAL_SAMPLES
        val_df = val_df.head(limit)

    # 2. Create Datasets
    train_dataset = DenoisingDataset(
        train_df, train_imgs, patch_size=patch_size, mode="train", augment=True
    )

    val_dataset = DenoisingDataset(
        val_df, val_imgs, patch_size=patch_size, mode="val", augment=False
    )

    # 3. Create DataLoaders
    def worker_init_fn(worker_id):
        # Ensure diverse seeds per worker, but reproducible per run
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random

        random.seed(worker_seed)

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(Config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=g,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=g,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates a DataLoader for the test set (full images, batch_size=1).
    """
    test_df, test_imgs = load_images(
        Config.TEST_METADATA, "test_cache", load_cached_data
    )

    # For test, we process full images one by one (batch_size=1)
    # patch_size=None indicates full image return
    test_dataset = DenoisingDataset(
        test_df, test_imgs, patch_size=None, mode="test", augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader
