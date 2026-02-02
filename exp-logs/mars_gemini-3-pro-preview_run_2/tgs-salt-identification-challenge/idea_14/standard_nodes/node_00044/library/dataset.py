import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Constants for the dataset
CACHE_DIR = "./working/idea_14/"
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"


def get_transforms(mode="train"):
    """
    Returns the Albumentations composition for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    # Common transforms: Padding to 128x128 and Normalization
    # We use reflection padding to handle the 101 -> 128 resizing smoothly
    common_transforms = [
        A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_REFLECT),
        # Normalize 1-channel image. Using Channel 0 of ImageNet stats as a proxy for grayscale intensity.
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ]

    if mode == "train":
        # Augmentations for training: Elastic and Rigid transformations
        train_transforms = [
            A.Compose(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                    ),
                ]
                + common_transforms
            )
        ]
        return train_transforms[0]
    else:
        # Validation/Test: Deterministic preprocessing
        return A.Compose(common_transforms)


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.
    Handles caching, depth normalization, Bernoulli masking, and augmentations.
    """

    def __init__(
        self,
        mode="train",
        df=None,
        depth_stats=None,
        load_cached_data=True,
        bernoulli_mask_prob=0.0,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            df (pd.DataFrame, optional): DataFrame containing dataset metadata.
                                         If None, loads from default metadata CSVs.
            depth_stats (tuple, optional): (mean, std) for depth normalization.
                                           If None, calculated from the current dataset (only for train).
            load_cached_data (bool): Whether to try loading data from numpy cache.
            bernoulli_mask_prob (float): Probability of masking depth (setting to 0) during training.
        """
        self.mode = mode
        self.bernoulli_mask_prob = bernoulli_mask_prob

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load Metadata if not provided
        if df is None:
            csv_path = os.path.join(METADATA_DIR, f"{mode}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
        else:
            self.df = df.reset_index(drop=True)

        # Handle Depth Statistics
        # We calculate stats on the current DF if not provided.
        # Ideally, pass training stats to val/test datasets.
        if depth_stats is None:
            self.depth_mean = self.df["z"].mean()
            self.depth_std = self.df["z"].std()
        else:
            self.depth_mean, self.depth_std = depth_stats

        # Load Data (Images, Masks, Depths)
        self._load_data(load_cached_data)

        # Setup Transforms
        self.transform = get_transforms(mode)

    def _load_data(self, load_cached_data):
        """
        Loads images and masks into memory, using caching to speed up subsequent runs.
        """
        # Define cache filenames based on mode (or a hash if we wanted to be more robust,
        # but mode is sufficient for this controlled environment)
        # If a custom DF is passed, we might overwrite 'train' cache, so we append a simple check
        # For this implementation, we assume standard modes or distinct cache prefixes.
        cache_prefix = self.mode

        img_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
        mask_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_masks.npy")
        depth_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_depths.npy")
        id_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

        # Attempt to load from cache
        if load_cached_data and os.path.exists(img_cache_path):
            try:
                self.images = np.load(img_cache_path)
                self.depths = np.load(depth_cache_path)
                self.ids = np.load(id_cache_path, allow_pickle=True)

                # Masks might not exist for test set
                if os.path.exists(mask_cache_path):
                    self.masks = np.load(mask_cache_path)
                else:
                    self.masks = None

                # Verify length consistency
                if len(self.images) == len(self.df):
                    return  # Loaded successfully
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from disk.")

        # Load from disk
        images = []
        masks = []
        depths = []
        ids = []

        has_masks = "mask_path" in self.df.columns

        for idx, row in self.df.iterrows():
            # Load Image
            img_path = os.path.join(INPUT_DIR, row["image_path"])
            # Load as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            images.append(img)

            # Load Mask if available
            if has_masks and pd.notna(row["mask_path"]):
                mask_path = os.path.join(INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # Fallback for empty masks if file missing but path exists?
                    # Should not happen per verification.
                    mask = np.zeros_like(img)
                # Binarize: 0 or 255 -> 0 or 1
                mask = (mask > 127).astype(np.uint8)
                masks.append(mask)

            # Depth
            depths.append(row["z"])
            ids.append(row["id"])

        # Convert to numpy arrays
        self.images = np.array(images, dtype=np.uint8)  # (N, 101, 101)
        self.depths = np.array(depths, dtype=np.float32)
        self.ids = np.array(ids)

        if has_masks and len(masks) > 0:
            self.masks = np.array(masks, dtype=np.uint8)  # (N, 101, 101)
        else:
            self.masks = None

        # Save to cache
        np.save(img_cache_path, self.images)
        np.save(depth_cache_path, self.depths)
        np.save(id_cache_path, self.ids)
        if self.masks is not None:
            np.save(mask_cache_path, self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get Image
        img = self.images[idx]  # (101, 101)

        # Expand dims for Albumentations (H, W) -> (H, W, 1)
        # This ensures ToTensorV2 produces (1, H, W)
        img = np.expand_dims(img, axis=2)

        # 2. Get Mask
        if self.masks is not None:
            mask = self.masks[idx]  # (101, 101)
            # Expand dims for Albumentations?
            # Albumentations expects mask to be (H, W) usually, but ToTensorV2 handles it.
            # However, for consistency with image, let's keep it (H, W) or (H, W, 1).
            # A.Compose handles 'image' and 'mask'.
        else:
            # Create dummy mask for test set
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # 3. Apply Augmentations
        # Albumentations expects dict
        augmented = self.transform(image=img, mask=mask)
        img_tensor = augmented["image"]  # (1, 128, 128)
        mask_tensor = augmented[
            "mask"
        ]  # (128, 128) or (1, 128, 128) depending on setup

        # Ensure mask is float tensor (BCE expects float) and has channel dim if needed
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, 128, 128)
        mask_tensor = mask_tensor.float()

        # 4. Process Depth
        z = self.depths[idx]

        # Normalize
        z_norm = (z - self.depth_mean) / (self.depth_std + 1e-8)

        # Bernoulli Scalar Masking
        # If training, randomly drop depth info to force learning from texture
        if self.mode == "train" and self.bernoulli_mask_prob > 0:
            if np.random.rand() < self.bernoulli_mask_prob:
                z_norm = 0.0  # Replace with mean (which is 0 after normalization)

        # Convert to tensor
        z_tensor = torch.tensor([z_norm], dtype=torch.float32)

        # 5. Get ID
        img_id = self.ids[idx]

        return img_tensor, mask_tensor, z_tensor, img_id
