import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import load_metadata, load_image_with_cache


class DenoisingDataset(Dataset):
    """
    Dataset for paired noisy and clean images with high-density sampling and caching.
    """

    def __init__(
        self,
        split: str,
        root_dir: str,
        cache_dir: str,
        patch_size: int = 128,
        samples_per_epoch: int = 1,
        augment: bool = False,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            root_dir (str): Path to the input directory (e.g., './input').
            cache_dir (str): Path to store cached .npy files.
            patch_size (int): Size of the square crop (H=W).
            samples_per_epoch (int): Multiplier for dataset length to enable high-density sampling.
            augment (bool): Whether to apply geometric augmentations.
        """
        self.split = split
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment

        # Load metadata
        self.metadata = load_metadata(split)
        self.ids = self.metadata["id"].tolist()
        self.feature_paths = self.metadata["feature_path"].tolist()

        # Check for labels (train/val have them, test does not)
        if "label_path" in self.metadata.columns:
            self.label_paths = self.metadata["label_path"].tolist()
        else:
            self.label_paths = None

    def __len__(self):
        # Virtual length: number of unique images * samples per image per epoch
        return len(self.ids) * self.samples_per_epoch

    def __getitem__(self, idx):
        # Map virtual index to actual image index
        img_idx = idx % len(self.ids)
        img_id = self.ids[img_idx]

        # 1. Load Images (with Caching)
        # Construct cache directory: ./working/idea_10/cache/{split}/
        split_cache_dir = os.path.join(self.cache_dir, self.split)

        # Load Noisy Image
        feature_rel_path = self.feature_paths[img_idx]
        feature_full_path = os.path.join(self.root_dir, feature_rel_path)
        noisy_cache_path = os.path.join(split_cache_dir, f"{img_id}_noisy.npy")

        noisy_img = load_image_with_cache(feature_full_path, noisy_cache_path)

        # Load Clean Image (if available)
        clean_img = None
        if self.label_paths:
            label_rel_path = self.label_paths[img_idx]
            label_full_path = os.path.join(self.root_dir, label_rel_path)
            clean_cache_path = os.path.join(split_cache_dir, f"{img_id}_clean.npy")
            clean_img = load_image_with_cache(label_full_path, clean_cache_path)

        # 2. Handle Dimensions & Padding
        h, w = noisy_img.shape

        # Pad if image is smaller than patch size
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h > 0 or pad_w > 0:
            # Reflect padding handles borders gracefully
            noisy_img = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode="reflect")
            if clean_img is not None:
                clean_img = np.pad(clean_img, ((0, pad_h), (0, pad_w)), mode="reflect")
            # Update dimensions
            h, w = noisy_img.shape

        # 3. Crop Extraction
        if self.augment:
            # Random Crop for Training
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            # Center Crop for Validation/Deterministic evaluation
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        noisy_patch = noisy_img[
            top : top + self.patch_size, left : left + self.patch_size
        ]
        if clean_img is not None:
            clean_patch = clean_img[
                top : top + self.patch_size, left : left + self.patch_size
            ]

        # 4. Geometric Augmentations
        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                noisy_patch = np.flip(noisy_patch, axis=1)
                if clean_img is not None:
                    clean_patch = np.flip(clean_patch, axis=1)

            # Random Vertical Flip
            if np.random.rand() < 0.5:
                noisy_patch = np.flip(noisy_patch, axis=0)
                if clean_img is not None:
                    clean_patch = np.flip(clean_patch, axis=0)

            # Random 90-degree Rotations
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                if clean_img is not None:
                    clean_patch = np.rot90(clean_patch, k)

        # 5. Convert to Tensor
        # Add channel dimension: (H, W) -> (1, H, W)
        # Use .copy() to ensure negative strides from flips are handled
        noisy_tensor = torch.from_numpy(noisy_patch.copy()).float().unsqueeze(0)

        if clean_img is not None:
            clean_tensor = torch.from_numpy(clean_patch.copy()).float().unsqueeze(0)
            return noisy_tensor, clean_tensor
        else:
            return noisy_tensor


def get_dataloaders(
    data_dir: str,
    cache_dir: str,
    batch_size: int,
    num_workers: int,
    patch_size: int = 128,
    train_samples_per_epoch: int = 100,
    val_samples_per_epoch: int = 1,
):
    """
    Creates and returns training and validation DataLoaders.

    Args:
        data_dir (str): Root directory of input data.
        cache_dir (str): Directory to store cached numpy files.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        patch_size (int): Size of image patches.
        train_samples_per_epoch (int): Multiplier for training set size (High-Density Sampling).
        val_samples_per_epoch (int): Multiplier for validation set size.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Train Dataset
    train_ds = DenoisingDataset(
        split="train",
        root_dir=data_dir,
        cache_dir=cache_dir,
        patch_size=patch_size,
        samples_per_epoch=train_samples_per_epoch,
        augment=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Dataset
    val_ds = DenoisingDataset(
        split="val",
        root_dir=data_dir,
        cache_dir=cache_dir,
        patch_size=patch_size,
        samples_per_epoch=val_samples_per_epoch,
        augment=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
