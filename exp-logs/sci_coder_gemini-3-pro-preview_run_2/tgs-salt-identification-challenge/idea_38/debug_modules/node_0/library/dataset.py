import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dataset_with_cache


def get_transforms(mode="train"):
    """
    Constructs the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', 'test', or 'pseudo'.

    Returns:
        A.Compose: The composition of transforms.
    """
    # Standard ImageNet statistics for normalization.
    # Since input is 1-channel (Grayscale), we use the first channel stats.
    mean = (0.485,)
    std = (0.229,)

    if mode == "train":
        return A.Compose(
            [
                # Non-Rigid Augmentation: Elastic Transform
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_ALPHA * 0.03,
                    p=Config.AUG_PROB,
                ),
                # Rigid Augmentation: Shift, Scale, Rotate
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_PROB,
                ),
                A.HorizontalFlip(p=0.5),
                # Normalization and Tensor Conversion
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Pseudo: Only Normalize and Convert
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles loading, padding, normalization, and augmentation.
    """

    def __init__(
        self,
        df,
        mode="train",
        cache_name="data",
        load_cached_data=True,
        subset_size=None,
        soft_masks=None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): Mode of operation ('train', 'val', 'test', 'pseudo').
            cache_name (str): Unique identifier for cache files.
            load_cached_data (bool): Whether to attempt loading from cache.
            subset_size (int, optional): Limit dataset size for debugging.
            soft_masks (np.array, optional): Soft probability masks for pseudo-labeling.
                                             Shape (N, 1, H, W). If provided, overrides GT masks.
        """
        self.mode = mode
        self.df = df

        # Load data using the utility function (handles padding to 128x128)
        data = load_dataset_with_cache(
            df,
            cache_name=cache_name,
            load_cached_data=load_cached_data,
            subset_size=subset_size,
        )

        self.images = data["images"]  # Shape: (N, 1, H, W), Values: [0, 1]
        self.depths = data["depths"]  # Shape: (N,)
        self.ids = data["ids"]  # Shape: (N,)

        # Handle masks: Use soft_masks if provided (Stage 3), else use loaded masks (Stage 1)
        if soft_masks is not None:
            self.masks = soft_masks
        else:
            self.masks = data["masks"]  # Shape: (N, 1, H, W) or None

        # Calculate depth statistics for Standard Scaling (z-score)
        # We compute stats on the currently loaded subset/split.
        self.depth_mean = np.mean(self.depths)
        self.depth_std = np.std(self.depths) + 1e-8  # Epsilon for stability

        # Initialize transforms
        self.transforms = get_transforms(mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Image Preparation
        # Transpose from (1, H, W) to (H, W, 1) for Albumentations
        image = self.images[idx].transpose(1, 2, 0)

        # 2. Mask Preparation
        mask = None
        if self.masks is not None:
            # Transpose from (1, H, W) to (H, W, 1)
            mask = self.masks[idx].transpose(1, 2, 0)

        # 3. Apply Augmentations
        if mask is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Ensure mask is a float tensor (N, H, W) -> (1, H, W) is handled by ToTensorV2 usually,
            # but we ensure float dtype for BCE/Lovasz loss compatibility.
            mask = mask.float()
        else:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 4. Depth Normalization (Standard Scaling)
        depth_val = self.depths[idx]
        depth_norm = (depth_val - self.depth_mean) / self.depth_std
        depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

        # 5. Return Data based on Mode
        if self.mode == "test":
            # Test: Image, Depth, ID
            return image, depth_tensor, self.ids[idx]

        elif self.mode == "pseudo":
            # Pseudo: Image, Soft Mask (Target)
            return image, mask

        else:
            # Train/Val: Image, Mask (GT), Depth
            return image, mask, depth_tensor
