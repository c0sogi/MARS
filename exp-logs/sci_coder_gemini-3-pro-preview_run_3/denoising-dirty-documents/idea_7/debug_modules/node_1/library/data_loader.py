import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_image_grayscale


def extract_patches(img_tensor, patch_size, stride):
    """
    Extracts patches from a single image tensor (1, H, W).
    Returns a numpy array of shape (N, 1, patch_size, patch_size).
    """
    # Convert tensor to numpy (H, W)
    img = img_tensor.squeeze(0).numpy()
    h, w = img.shape

    patches = []

    # Extract patches with overlap
    # Range end is exclusive, so we add 1 to ensure the last valid start index is covered
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    if not patches:
        return np.empty((0, 1, patch_size, patch_size), dtype=np.float32)

    # Stack patches and add channel dimension: (N, H, W) -> (N, 1, H, W)
    return np.array(patches, dtype=np.float32)[:, np.newaxis, :, :]


def process_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads images from metadata, calculates noise targets, extracts patches,
    and caches the result as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (patches_array, targets_array)
    """
    patches_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_patches.npy")
    targets_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_targets.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(patches_cache_path) and os.path.exists(targets_cache_path):
            print(f"Loading cached {cache_prefix} data from {Config.WORKING_DIR}...")
            try:
                patches = np.load(patches_cache_path)
                targets = np.load(targets_cache_path)
                return patches, targets
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data.")

    # 2. Process from scratch
    print(f"Processing {cache_prefix} data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_patches = []
    all_targets = []

    for _, row in df.iterrows():
        input_rel_path = row["input_path"]
        target_rel_path = row["target_path"]

        input_full_path = os.path.join(Config.INPUT_DIR, input_rel_path)
        target_full_path = os.path.join(Config.INPUT_DIR, target_rel_path)

        # Load images as tensors (1, H, W)
        noisy_tensor = load_image_grayscale(input_full_path)
        clean_tensor = load_image_grayscale(target_full_path)

        # Calculate Noise Target: Noise = Input - Clean
        noise_target_tensor = noisy_tensor - clean_tensor

        # Extract patches
        img_patches = extract_patches(noisy_tensor, Config.PATCH_SIZE, Config.STRIDE)
        tgt_patches = extract_patches(
            noise_target_tensor, Config.PATCH_SIZE, Config.STRIDE
        )

        all_patches.append(img_patches)
        all_targets.append(tgt_patches)

    # Concatenate all patches into a single array
    if all_patches:
        patches_arr = np.concatenate(all_patches, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
    else:
        patches_arr = np.empty(
            (0, 1, Config.PATCH_SIZE, Config.PATCH_SIZE), dtype=np.float32
        )
        targets_arr = np.empty(
            (0, 1, Config.PATCH_SIZE, Config.PATCH_SIZE), dtype=np.float32
        )

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(patches_cache_path, patches_arr)
    np.save(targets_cache_path, targets_arr)
    print(f"Saved {cache_prefix} data: {patches_arr.shape} patches.")

    return patches_arr, targets_arr


class DenoisingDataset(Dataset):
    """
    Dataset for training the denoising model.
    Returns pairs of (noisy_patch, noise_target).
    """

    def __init__(self, patches, targets, augment=False):
        self.patches = patches
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Load data
        patch_np = self.patches[idx]
        target_np = self.targets[idx]

        # Convert to tensor
        patch = torch.from_numpy(patch_np)
        target = torch.from_numpy(target_np)

        # Apply Augmentation
        if self.augment:
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [2])
                target = torch.flip(target, [2])

            # Random Vertical Flip
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, [1])
                target = torch.flip(target, [1])

            # Random Rotation (0, 90, 180, 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch = torch.rot90(patch, k, [1, 2])
                target = torch.rot90(target, k, [1, 2])

        return patch, target


class TestDataset(Dataset):
    """
    Dataset for inference on the test set.
    Returns full images and their IDs.
    """

    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["image_id"]).split(".")[0]  # Remove extension
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

        # Load full image
        img = load_image_grayscale(input_path)

        return img, img_id


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached patch data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Prepare Training Data
    train_patches, train_targets = process_data(
        Config.TRAIN_METADATA, "train", load_cached_data=load_cached_data
    )

    # Prepare Validation Data
    val_patches, val_targets = process_data(
        Config.VAL_METADATA, "val", load_cached_data=load_cached_data
    )

    # Create Datasets
    # Enable augmentation only for training
    train_dataset = DenoisingDataset(train_patches, train_targets, augment=True)
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates a DataLoader for the test set.
    """
    test_dataset = TestDataset(Config.TEST_METADATA)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one full image at a time
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return test_loader
