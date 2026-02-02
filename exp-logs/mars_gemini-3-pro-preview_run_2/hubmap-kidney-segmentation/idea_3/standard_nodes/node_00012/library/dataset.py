import os
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode, polygons_to_mask


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    """
    if phase == "train":
        # Spatial transforms applied to both image (4ch) and mask
        spatial_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
            ]
        )

        # Color transforms applied only to RGB channels
        color_transforms = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
                ),
                A.RandomGamma(p=0.2),
            ]
        )

        return spatial_transforms, color_transforms
    else:
        return None, None


def load_and_preprocess_data(
    metadata_path, cache_dir, load_cached_data=True, is_train=True
):
    """
    Loads data from metadata, processes it into tiles (Context-Fused Sampling),
    and caches the result as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_dir (str): Directory to save/load cached .npy files.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_train (bool): If True, applies undersampling to background tiles.

    Returns:
        tuple: (images_np, masks_np)
    """
    os.makedirs(cache_dir, exist_ok=True)
    images_cache_path = os.path.join(cache_dir, "images.npy")
    masks_cache_path = os.path.join(cache_dir, "masks.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(masks_cache_path)
    ):
        print(f"Loading cached data from {cache_dir}...")
        try:
            images = np.load(images_cache_path)
            masks = np.load(masks_cache_path)
            return images, masks
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df = df.head(1)  # Process only 1 image in debug mode

    all_tiles = []
    all_masks = []

    tile_size = Config.TILE_SIZE

    for idx, row in df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        anat_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])

        # Load Image (RGB)
        try:
            with rasterio.open(img_path) as src:
                # Read all bands and transpose to (H, W, C)
                image = src.read().transpose(1, 2, 0)

                # Standardize to 3 channels (RGB)
                if image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
                elif image.shape[2] > 3:
                    image = image[:, :, :3]
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue

        h, w = image.shape[:2]

        # Load Glomerulus Mask (Target)
        if "encoding" in row and pd.notna(row["encoding"]):
            mask = rle_decode(row["encoding"], (h, w))
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Load Anatomical Mask (Context Channel)
        # We target the 'Cortex' region
        anat_mask = polygons_to_mask(anat_path, (h, w), label_name="Cortex")

        # Create 4-channel input: RGB + Anatomical
        # Expand anat_mask to (H, W, 1)
        anat_mask_exp = np.expand_dims(anat_mask, axis=-1)

        # Concatenate (Result is uint8, anat_mask is 0/1)
        combined_img = np.concatenate([image, anat_mask_exp], axis=2)

        # Pad image to be multiple of tile_size
        pad_h = (tile_size - (h % tile_size)) % tile_size
        pad_w = (tile_size - (w % tile_size)) % tile_size

        if pad_h > 0 or pad_w > 0:
            combined_img = np.pad(
                combined_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        new_h, new_w = combined_img.shape[:2]

        # Generate Tiles
        for y in range(0, new_h, tile_size):
            for x in range(0, new_w, tile_size):
                img_tile = combined_img[y : y + tile_size, x : x + tile_size, :]
                mask_tile = mask[y : y + tile_size, x : x + tile_size]

                # Verify shape
                if img_tile.shape[0] != tile_size or img_tile.shape[1] != tile_size:
                    continue

                # Context-Fused Representative Sampling
                has_glomerulus = mask_tile.sum() > 0

                if is_train:
                    if has_glomerulus:
                        # Keep 100% of positives
                        all_tiles.append(img_tile)
                        all_masks.append(mask_tile)
                    else:
                        # Sample background tiles
                        if np.random.rand() < Config.BACKGROUND_SAMPLING_RATIO:
                            all_tiles.append(img_tile)
                            all_masks.append(mask_tile)
                else:
                    # For validation, keep all tiles to ensure unbiased metric calculation
                    # (or apply same sampling if memory is constrained, but here we keep all)
                    all_tiles.append(img_tile)
                    all_masks.append(mask_tile)

    # Convert to numpy arrays
    if len(all_tiles) == 0:
        print("Warning: No tiles generated. Creating dummy batch.")
        images_np = np.zeros((1, tile_size, tile_size, 4), dtype=np.uint8)
        masks_np = np.zeros((1, tile_size, tile_size), dtype=np.uint8)
    else:
        images_np = np.array(all_tiles, dtype=np.uint8)
        masks_np = np.array(all_masks, dtype=np.uint8)

    # Cache results
    print(f"Saving {images_np.shape[0]} tiles to cache at {cache_dir}...")
    np.save(images_cache_path, images_np)
    np.save(masks_cache_path, masks_np)

    return images_np, masks_np


class HuBMAPDataset(Dataset):
    def __init__(self, images, masks, phase="train"):
        self.images = images
        self.masks = masks
        self.phase = phase
        self.spatial_aug, self.color_aug = get_transforms(phase)

        # Normalization constants for RGB (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 4)
        mask = self.masks[idx]  # (H, W)

        # 1. Spatial Augmentation (Jointly on 4-channel img and mask)
        if self.spatial_aug:
            augmented = self.spatial_aug(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 2. Split Channels
        rgb = image[..., :3]  # (H, W, 3)
        anatomy = image[..., 3]  # (H, W)

        # 3. Color Augmentation (RGB only)
        if self.color_aug:
            rgb = self.color_aug(image=rgb)["image"]

        # 4. Normalization
        # RGB: [0, 255] -> [0, 1] -> (x - mean) / std
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std

        # Anatomy: [0, 1] (uint8) -> [0.0, 1.0] (float)
        # Expand dims for anatomy to match (H, W, 1)
        anatomy = anatomy.astype(np.float32)
        anatomy = np.expand_dims(anatomy, axis=-1)

        # 5. Recombine
        # Result: (H, W, 4)
        image_tensor = np.concatenate([rgb, anatomy], axis=-1)

        # 6. To Tensor (Channel First)
        # (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float()

        # Mask: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0).float()

        return image_tensor, mask_tensor


def get_dataloader(
    phase,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Factory function to create dataloaders for training or validation.
    """
    is_train = phase == "train"

    if phase == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_dir = Config.TRAIN_CACHE_DIR
    elif phase == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_dir = Config.VAL_CACHE_DIR
    else:
        raise ValueError("Phase must be 'train' or 'val'")

    images, masks = load_and_preprocess_data(
        metadata_path, cache_dir, load_cached_data, is_train
    )

    dataset = HuBMAPDataset(images, masks, phase=phase)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )
