import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import pad_image, load_data_with_cache

# Global Statistics from EDA
# Mean: 148.07, Std: 65.21 (on 0-255 scale)
# Converted to 0-1 scale for Albumentations/Pytorch
PIXEL_MEAN = 148.07 / 255.0
PIXEL_STD = 65.21 / 255.0


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles images, masks (binary or soft), and depth values.
    """

    def __init__(
        self,
        ids,
        images,
        depths,
        masks=None,
        mode="train",
        transform=None,
        depth_mean=0.0,
        depth_std=1.0,
    ):
        """
        Args:
            ids (np.ndarray): Array of image IDs.
            images (np.ndarray): Array of images (N, H, W).
            depths (np.ndarray): Array of depth values (N,).
            masks (np.ndarray, optional): Array of masks (N, H, W).
            mode (str): 'train', 'val', 'test', or 'pseudo'.
            transform (A.Compose): Albumentations transform pipeline.
            depth_mean (float): Mean of depth for normalization.
            depth_std (float): Std of depth for normalization.
        """
        self.ids = ids
        self.images = images
        self.depths = depths
        self.masks = masks
        self.mode = mode
        self.transform = transform
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Retrieve raw data
        # Images are (101, 101) uint8
        image = self.images[idx]
        depth = self.depths[idx]

        # 2. Pad Image (Reflection Padding) -> (128, 128)
        image = pad_image(image)

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            mask = pad_image(mask)

        # 3. Prepare for Albumentations (H, W) -> (H, W, C)
        # We add a channel dimension because Albumentations expects it for some transforms
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)

        # 4. Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # 5. Process Depth (Standard Scaling)
        depth_norm = (depth - self.depth_mean) / self.depth_std
        depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

        # 6. Return Data based on Mode
        if self.mode in ["train", "val", "pseudo"]:
            # Handle Mask Tensor
            # If ToTensorV2 was used, mask is (C, H, W) Tensor.
            # If mask is binary, it should be float for BCE/Lovasz loss.
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)

            mask = mask.float()

            # Ensure shape is (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                # If Albumentations returned (H, W, 1), permute to (1, H, W)
                # Note: ToTensorV2 usually handles this, but safety check
                if mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)

            return image, mask, depth_tensor

        elif self.mode == "test":
            # Return ID for submission generation
            return image, depth_tensor, self.ids[idx]

        return image


def _load_images_from_df(df, input_root):
    """Helper to load all images from a dataframe into a numpy array."""
    images = []
    for path in df["image_path"]:
        full_path = os.path.join(input_root, path)
        # Load as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for safety, though metadata check passed
            img = np.zeros((Config.ORIG_H, Config.ORIG_W), dtype=np.uint8)
        images.append(img)
    return np.array(images, dtype=np.uint8)


def _load_masks_from_df(df, input_root):
    """Helper to load all masks from a dataframe into a numpy array."""
    masks = []
    for path in df["mask_path"]:
        full_path = os.path.join(input_root, path)
        msk = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.zeros((Config.ORIG_H, Config.ORIG_W), dtype=np.uint8)
        # Ensure binary (0 or 1)
        msk = (msk > 127).astype(np.uint8)
        masks.append(msk)
    return np.array(masks, dtype=np.uint8)


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Factory function to create DataLoaders with caching and preprocessing.
    """

    # 1. Load Metadata
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)
    test_df = pd.read_csv(test_metadata_path)

    if debug:
        print(f"Debug mode: Reducing dataset size to {Config.DEBUG_SAMPLE_SIZE}")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 2. Define Caching Logic
    # We define closures to pass to the generic caching function
    def load_train_images():
        return _load_images_from_df(train_df, Config.INPUT_ROOT)

    def load_train_masks():
        return _load_masks_from_df(train_df, Config.INPUT_ROOT)

    def load_val_images():
        return _load_images_from_df(val_df, Config.INPUT_ROOT)

    def load_val_masks():
        return _load_masks_from_df(val_df, Config.INPUT_ROOT)

    def load_test_images():
        return _load_images_from_df(test_df, Config.INPUT_ROOT)

    cache_root = Config.CACHE_DIR

    # 3. Load Data (from Cache or Compute)
    # Note: If debug is True, we might re-cache small files or just overwrite.
    # For simplicity, we use unique names or rely on the user to clear cache if debugging changes.
    # Here we assume cache is for full dataset. If debug, we skip cache loading to avoid size mismatch.
    use_cache = load_cached_data and not debug

    train_images = load_data_with_cache(
        os.path.join(cache_root, "train_images.npy"), load_train_images, use_cache
    )
    train_masks = load_data_with_cache(
        os.path.join(cache_root, "train_masks.npy"), load_train_masks, use_cache
    )
    val_images = load_data_with_cache(
        os.path.join(cache_root, "val_images.npy"), load_val_images, use_cache
    )
    val_masks = load_data_with_cache(
        os.path.join(cache_root, "val_masks.npy"), load_val_masks, use_cache
    )
    test_images = load_data_with_cache(
        os.path.join(cache_root, "test_images.npy"), load_test_images, use_cache
    )

    # Load Depths directly (fast enough)
    train_depths = train_df["z"].values.astype(np.float32)
    val_depths = val_df["z"].values.astype(np.float32)
    test_depths = test_df["z"].values.astype(np.float32)

    # 4. Calculate Depth Statistics (from Training Set)
    depth_mean = train_depths.mean()
    depth_std = train_depths.std() + 1e-6  # Avoid div by zero

    # 5. Define Transforms
    # Albumentations Normalize: (x/255 - mean) / std
    # ToTensorV2: Converts HWC to CHW and to Tensor

    train_transform = A.Compose(
        [
            A.ElasticTransform(
                alpha=Config.AUG_ELASTIC_ALPHA,
                sigma=Config.AUG_ELASTIC_SIGMA,
                alpha_affine=Config.AUG_ELASTIC_ALPHA_AFFINE,
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=Config.AUG_SHIFT_SCALE_ROTATE_P,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(PIXEL_MEAN,), std=(PIXEL_STD,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(PIXEL_MEAN,), std=(PIXEL_STD,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # 6. Instantiate Datasets
    train_dataset = SaltDataset(
        ids=train_df["id"].values,
        images=train_images,
        depths=train_depths,
        masks=train_masks,
        mode="train",
        transform=train_transform,
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    val_dataset = SaltDataset(
        ids=val_df["id"].values,
        images=val_images,
        depths=val_depths,
        masks=val_masks,
        mode="val",
        transform=val_transform,
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    test_dataset = SaltDataset(
        ids=test_df["id"].values,
        images=test_images,
        depths=test_depths,
        masks=None,
        mode="test",
        transform=val_transform,
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
