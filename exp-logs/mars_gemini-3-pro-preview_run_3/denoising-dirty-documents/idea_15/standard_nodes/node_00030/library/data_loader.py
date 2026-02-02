import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def extract_dense_patches(
    metadata_path, cache_patches_file, cache_targets_file, load_cached_data=True
):
    """
    Extracts high-density overlapping patches from images listed in the metadata.
    Handles caching to disk to save time on subsequent runs.
    """
    patches_path = os.path.join(Config.WORKING_DIR, cache_patches_file)
    targets_path = os.path.join(Config.WORKING_DIR, cache_targets_file)

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(patches_path)
        and os.path.exists(targets_path)
    ):
        print(f"Loading cached patches from {patches_path}...")
        try:
            patches = np.load(patches_path)
            targets = np.load(targets_path)
            print(f"Loaded {len(patches)} patches.")
            return patches, targets
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting patches from images in {metadata_path}...")
    df = pd.read_csv(metadata_path)

    patch_size = Config.PATCH_SIZE
    stride = Config.STRIDE

    patches_list = []
    targets_list = []

    for _, row in df.iterrows():
        # Load images
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        target_path = os.path.join(Config.INPUT_DIR, row["target_path"])

        # Read as grayscale
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if img_in is None or img_tar is None:
            continue

        # Normalize to [0, 1]
        img_in = img_in.astype(np.float32) / 255.0
        img_tar = img_tar.astype(np.float32) / 255.0

        h, w = img_in.shape

        # Extract patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_in = img_in[y : y + patch_size, x : x + patch_size]
                patch_tar = img_tar[y : y + patch_size, x : x + patch_size]

                patches_list.append(patch_in)
                targets_list.append(patch_tar)

    # Convert to numpy arrays (N, 1, H, W)
    # We add the channel dimension here
    patches = np.array(patches_list)[:, np.newaxis, :, :]
    targets = np.array(targets_list)[:, np.newaxis, :, :]

    print(f"Extracted {len(patches)} patches. Saving to cache...")

    # 3. Save to cache
    np.save(patches_path, patches)
    np.save(targets_path, targets)

    return patches, targets


class DenoisingDataset(Dataset):
    """
    Dataset for training/validation.
    Yields (noisy_patch, residual_noise_target).
    Applies rigid augmentations if enabled.
    """

    def __init__(self, patches, clean_targets, augment=False):
        self.patches = patches
        self.clean_targets = clean_targets
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Data is already (1, H, W) float32
        noisy = self.patches[idx].copy()
        clean = self.clean_targets[idx].copy()

        # Augmentation (Rigid: Flips and Rotations)
        if self.augment:
            # Random flip horizontal
            if np.random.rand() > 0.5:
                noisy = np.flip(noisy, axis=2)
                clean = np.flip(clean, axis=2)

            # Random flip vertical
            if np.random.rand() > 0.5:
                noisy = np.flip(noisy, axis=1)
                clean = np.flip(clean, axis=1)

            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k, axes=(1, 2))
                clean = np.rot90(clean, k, axes=(1, 2))

        # Ensure contiguous arrays after flipping/rotation for PyTorch
        noisy = np.ascontiguousarray(noisy)
        clean = np.ascontiguousarray(clean)

        # Calculate Residual (Target for the network)
        # Network predicts: Noise
        # Noise = Noisy_Image - Clean_Image
        residual = noisy - clean

        return torch.from_numpy(noisy), torch.from_numpy(residual)


class TestDataset(Dataset):
    """
    Dataset for Test Inference.
    Yields full images and their IDs.
    """

    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for safety, though shouldn't happen based on EDA
            img = np.zeros((50, 50), dtype=np.uint8)

        # Normalize
        img = img.astype(np.float32) / 255.0

        # Add channel dim: (1, H, W)
        img = img[np.newaxis, :, :]

        return torch.from_numpy(img), img_id


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create Train, Val, and Test DataLoaders.
    """
    seed_everything(Config.SEED)

    # --- Train Data ---
    train_patches, train_targets = extract_dense_patches(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_PATCHES,
        Config.CACHE_TRAIN_TARGETS,
        load_cached_data=load_cached_data,
    )

    train_dataset = DenoisingDataset(
        train_patches, train_targets, augment=Config.AUGMENTATION
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # --- Validation Data ---
    val_patches, val_targets = extract_dense_patches(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_PATCHES,
        Config.CACHE_VAL_TARGETS,
        load_cached_data=load_cached_data,
    )

    val_dataset = DenoisingDataset(
        val_patches, val_targets, augment=False  # No augmentation for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # --- Test Data ---
    test_dataset = TestDataset(Config.TEST_METADATA_PATH)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one full image at a time
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
