import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_image, normalize_image, set_seed


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising task.
    Handles Training (patches + aug), Validation (full images), and Testing (full images).
    """

    def __init__(
        self,
        metadata_df,
        root_dir,
        cache_dir,
        mode="train",
        patches_per_image=1,
        patch_size=128,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata (id, feature_path, label_path).
            root_dir (str): Root directory for input images.
            cache_dir (str): Directory to store/load cached .npy files.
            mode (str): 'train', 'val', or 'test'.
            patches_per_image (int): Number of random patches to extract per image per epoch (only for train).
            patch_size (int): Size of the square patch to crop (only for train).
        """
        self.metadata = metadata_df
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.mode = mode
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size

        # Pre-calculate paths to avoid overhead in __getitem__
        self.data = []
        for _, row in self.metadata.iterrows():
            item = {
                "id": str(row["id"]),
                "feature_path": os.path.join(self.root_dir, row["feature_path"]),
                "feature_cache": os.path.join(
                    self.cache_dir, row["feature_path"].replace(".png", ".npy")
                ),
            }
            if "label_path" in row and pd.notna(row["label_path"]):
                item["label_path"] = os.path.join(self.root_dir, row["label_path"])
                item["label_cache"] = os.path.join(
                    self.cache_dir, row["label_path"].replace(".png", ".npy")
                )
            self.data.append(item)

    def __len__(self):
        if self.mode == "train":
            return len(self.data) * self.patches_per_image
        else:
            return len(self.data)

    def __getitem__(self, idx):
        # Map linear index to image index
        if self.mode == "train":
            img_idx = idx // self.patches_per_image
        else:
            img_idx = idx

        item = self.data[img_idx]

        # Load Noisy Image
        noisy_img = load_image(item["feature_path"], item["feature_cache"])
        noisy_img = normalize_image(noisy_img)

        # Load Clean Image (if available)
        clean_img = None
        if "label_path" in item:
            clean_img = load_image(item["label_path"], item["label_cache"])
            clean_img = normalize_image(clean_img)

        # --- Training Logic: Random Crop & Augmentation ---
        if self.mode == "train":
            h, w = noisy_img.shape

            # Random Crop
            # Ensure image is large enough, otherwise take what we have (though EDA confirms images are > 128)
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                # Pad if smaller than patch size (unlikely based on EDA)
                noisy_img = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean_img = np.pad(clean_img, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy_img.shape

            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy_img[
                top : top + self.patch_size, left : left + self.patch_size
            ]
            clean_patch = clean_img[
                top : top + self.patch_size, left : left + self.patch_size
            ]

            # Augmentations
            # 1. Random Horizontal Flip
            if np.random.rand() > 0.5:
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)

            # 2. Random Vertical Flip
            if np.random.rand() > 0.5:
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # 3. Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                clean_patch = np.rot90(clean_patch, k)

            # Convert to Tensor (C, H, W) -> (1, H, W)
            # Must use .copy() because numpy flips/rotates can create negative strides which torch doesn't support
            noisy_tensor = torch.from_numpy(noisy_patch.copy()).unsqueeze(0)
            clean_tensor = torch.from_numpy(clean_patch.copy()).unsqueeze(0)

            return noisy_tensor, clean_tensor

        # --- Validation Logic: Full Image ---
        elif self.mode == "val":
            # Returns full images for accurate metric calculation
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
            return noisy_tensor, clean_tensor, item["id"]

        # --- Test Logic: Full Image ---
        elif self.mode == "test":
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
            return noisy_tensor, item["id"]


def get_dataloaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    test_csv_path=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        train_csv_path (str): Path to training metadata CSV.
        val_csv_path (str): Path to validation metadata CSV.
        test_csv_path (str): Path to test metadata CSV.
        batch_size (int): Batch size for training.
        num_workers (int): Number of worker processes.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """

    # Load DataFrames
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Instantiate Datasets
    train_dataset = DenoisingDataset(
        metadata_df=df_train,
        root_dir=Config.INPUT_DIR,
        cache_dir=Config.CACHE_DIR,
        mode="train",
        patches_per_image=Config.PATCHES_PER_IMAGE,
        patch_size=Config.PATCH_SIZE,
    )

    val_dataset = DenoisingDataset(
        metadata_df=df_val,
        root_dir=Config.INPUT_DIR,
        cache_dir=Config.CACHE_DIR,
        mode="val",
    )

    test_dataset = DenoisingDataset(
        metadata_df=df_test,
        root_dir=Config.INPUT_DIR,
        cache_dir=Config.CACHE_DIR,
        mode="test",
    )

    # Create DataLoaders

    # Train Loader: Batched, Shuffled
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain consistent stats
    )

    # Val Loader: Batch Size 1 (Variable Image Sizes), No Shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Loader: Batch Size 1 (Variable Image Sizes), No Shuffle
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
