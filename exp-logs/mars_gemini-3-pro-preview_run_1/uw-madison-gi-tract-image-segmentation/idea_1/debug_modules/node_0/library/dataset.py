import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from library.utils import rle_decode


class UWMadisonDataset(Dataset):
    def __init__(
        self,
        mode="train",
        fraction=1.0,
        load_cached_data=True,
        img_size=256,
    ):
        """
        PyTorch Dataset for UW-Madison GI Tract Image Segmentation.

        Args:
            mode (str): One of 'train', 'val', 'test'.
            fraction (float): Fraction of the dataset to use (for debugging).
            load_cached_data (bool): Whether to load/save metadata from/to cache.
            img_size (int): Target image size for resizing (square).
        """
        self.mode = mode
        self.fraction = fraction
        self.img_size = img_size
        self.input_dir = "./input"
        self.cache_dir = "./working/idea_1"

        # Define transforms
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.Resize(height=img_size, width=img_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=img_size, width=img_size, p=1.0),
                ]
            )

        # Load metadata
        self.df = self._load_metadata(load_cached_data)

    def _load_metadata(self, load_cached_data):
        """
        Loads metadata with caching mechanism using Parquet.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"{self.mode}_metadata.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                pass  # Fallback to loading from source

        # 2. Load from source
        csv_path = os.path.join("./metadata", f"{self.mode}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        # keep_default_na=False prevents empty RLE strings from becoming NaN
        df = pd.read_csv(csv_path, keep_default_na=False)

        # Subsample if fraction < 1.0
        if self.fraction < 1.0:
            df = df.sample(frac=self.fraction, random_state=42).reset_index(drop=True)

        # 3. Save to cache
        if load_cached_data:
            df.to_parquet(cache_path, index=False)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image (16-bit grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback for safety, though metadata validation ensures existence
            img = np.zeros((row["height"], row["width"]), dtype=np.uint16)

        # Normalization: Min-Max scaling to [0, 1]
        # Cast to float32 for training
        img = img.astype(np.float32)
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img - min_val  # Should be all zeros

        # Prepare for Albumentations (H, W, C)
        # Grayscale to (H, W, 1) if needed, but albumentations handles 2D images too.
        # However, to keep it consistent with masks, we can treat it as 2D array.

        original_h = row["height"]
        original_w = row["width"]

        if self.mode in ["train", "val"]:
            # Load Masks
            # Classes: large_bowel, small_bowel, stomach
            # We stack them to create (H, W, 3)
            mask_shape = (original_h, original_w)

            mask_lb = rle_decode(row["large_bowel"], mask_shape)
            mask_sb = rle_decode(row["small_bowel"], mask_shape)
            mask_st = rle_decode(row["stomach"], mask_shape)

            # Stack: (H, W, 3)
            mask = np.stack([mask_lb, mask_sb, mask_st], axis=-1).astype(np.float32)

            # Apply Augmentations
            transformed = self.transform(image=img, mask=mask)
            img_aug = transformed["image"]
            mask_aug = transformed["mask"]

            # Convert to Tensor (C, H, W)
            # Image is currently (H, W) or (H, W, 1) depending on albumentations behavior with grayscale
            if img_aug.ndim == 2:
                img_aug = img_aug[np.newaxis, ...]  # (1, H, W)
            else:
                img_aug = img_aug.transpose(2, 0, 1)  # (C, H, W)

            # Mask is (H, W, 3) -> (3, H, W)
            mask_aug = mask_aug.transpose(2, 0, 1)

            return torch.from_numpy(img_aug), torch.from_numpy(mask_aug)

        else:
            # Test mode: No masks
            transformed = self.transform(image=img)
            img_aug = transformed["image"]

            if img_aug.ndim == 2:
                img_aug = img_aug[np.newaxis, ...]
            else:
                img_aug = img_aug.transpose(2, 0, 1)

            # Return ID and original shape for post-processing
            return (
                torch.from_numpy(img_aug),
                row["id"],
                np.array([original_h, original_w]),
            )
