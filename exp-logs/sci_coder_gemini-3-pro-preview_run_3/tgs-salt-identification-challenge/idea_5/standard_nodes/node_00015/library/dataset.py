import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import pad_image


def get_salt_data(metadata_path, mode, load_cached_data=True):
    """
    Loads dataset arrays from metadata or disk cache.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
            - images: (N, H, W, 3) uint8 array
            - masks: (N, H, W, 1) uint8 array (or None for test)
            - depths: (N,) float32 array
            - ids: (N,) string array
    """
    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    cache_images = os.path.join(cache_dir, f"cached_{mode}_images.npy")
    cache_masks = os.path.join(cache_dir, f"cached_{mode}_masks.npy")
    cache_depths = os.path.join(cache_dir, f"cached_{mode}_depths.npy")
    cache_ids = os.path.join(cache_dir, f"cached_{mode}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        # Check if essential files exist
        files_exist = (
            os.path.exists(cache_images)
            and os.path.exists(cache_depths)
            and os.path.exists(cache_ids)
        )

        # For train/val, mask cache must also exist
        if mode != "test" and not os.path.exists(cache_masks):
            files_exist = False

        if files_exist:
            print(f"Loading {mode} data from cache...")
            images = np.load(cache_images)
            depths = np.load(cache_depths)
            ids = np.load(cache_ids)

            if mode != "test":
                masks = np.load(cache_masks)
                return images, masks, depths, ids
            else:
                return images, None, depths, ids

    # Process data from scratch
    print(f"Processing {mode} data from scratch...")
    df = pd.read_csv(metadata_path)

    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        # Load as color (BGR) then convert to RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)

        # Load Mask (only for train/val)
        if mode != "test":
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            masks_list.append(mask)

        # Load Depth and ID
        depths_list.append(row["z"])
        ids_list.append(row["id"])

    # Convert to Numpy Arrays
    images = np.array(images_list, dtype=np.uint8)
    depths = np.array(depths_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to Cache
    np.save(cache_images, images)
    np.save(cache_depths, depths)
    np.save(cache_ids, ids)

    if mode != "test":
        masks = np.array(masks_list, dtype=np.uint8)
        # Ensure masks have channel dimension: (N, H, W) -> (N, H, W, 1)
        if masks.ndim == 3:
            masks = masks[..., np.newaxis]
        np.save(cache_masks, masks)
        return images, masks, depths, ids

    return images, None, depths, ids


class SaltDataset(Dataset):
    def __init__(
        self,
        mode="train",
        metadata_path=None,
        transform=None,
        load_cached_data=True,
        debug=False,
    ):
        """
        PyTorch Dataset for Salt Segmentation.

        Args:
            mode (str): 'train', 'val', or 'test'.
            metadata_path (str, optional): Path to metadata CSV. Defaults to Config paths.
            transform (albumentations.Compose, optional): Augmentation pipeline.
            load_cached_data (bool): Use cached .npy files if available.
            debug (bool): If True, limits dataset to 32 samples for debugging.
        """
        self.mode = mode
        self.transform = transform

        # Determine metadata path if not provided
        if metadata_path is None:
            if mode == "train":
                metadata_path = Config.TRAIN_METADATA
            elif mode == "val":
                metadata_path = Config.VAL_METADATA
            elif mode == "test":
                metadata_path = Config.TEST_METADATA

        # Load Data
        self.images, self.masks, self.depths, self.ids = get_salt_data(
            metadata_path, mode, load_cached_data
        )

        # Load Global Depth Statistics for Normalization
        # We read the full depths.csv to ensure global min/max consistency
        depth_df = pd.read_csv(Config.DEPTHS_CSV)
        self.min_depth = depth_df["z"].min()
        self.max_depth = depth_df["z"].max()

        # Debug Mode
        if debug:
            print(f"Debug mode enabled for {mode}: limiting to 32 samples.")
            self.images = self.images[:32]
            self.depths = self.depths[:32]
            self.ids = self.ids[:32]
            if self.masks is not None:
                self.masks = self.masks[:32]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Raw Data
        image = self.images[idx]  # (101, 101, 3) uint8
        depth_val = self.depths[idx]  # scalar float

        # 2. Normalize Depth (0 to 1)
        depth_norm = (depth_val - self.min_depth) / (self.max_depth - self.min_depth)

        # 3. Construct Depth Channel
        # Create a channel (H, W) filled with the normalized depth value
        h, w = image.shape[:2]
        depth_channel = np.full((h, w), depth_norm, dtype=np.float32)

        # 4. Normalize Image (0 to 1)
        # Convert RGB to float32 [0, 1]
        image = image.astype(np.float32) / 255.0

        # 5. Construct 3-Channel Input (Cite solution_lesson_node_00014)
        # Stack [Gray, Gray, Depth] to preserve pre-trained weights
        image_gray = image[:, :, 0]
        image_3c = np.dstack([image_gray, image_gray, depth_channel])

        # 6. Reflection Padding
        # Pad 101x101 to 128x128
        image_padded = pad_image(image_3c, target_size=Config.IMG_SIZE)

        mask_padded = None
        if self.mode != "test":
            mask = self.masks[idx]  # (101, 101, 1) uint8
            # Normalize mask to [0, 1] float
            mask = mask.astype(np.float32) / 255.0

            # Pad mask
            mask_padded = pad_image(mask, target_size=Config.IMG_SIZE)

            # Ensure channel dimension is preserved after padding
            if mask_padded.ndim == 2:
                mask_padded = mask_padded[..., np.newaxis]

        # 7. Apply Augmentations
        # Note: Albumentations handles multi-channel images (like 4-channel) correctly
        # for geometric transforms like HorizontalFlip.
        if self.transform:
            if self.mode != "test":
                augmented = self.transform(image=image_padded, mask=mask_padded)
                image_padded = augmented["image"]
                mask_padded = augmented["mask"]
            else:
                augmented = self.transform(image=image_padded)
                image_padded = augmented["image"]

        # 8. Convert to Tensor (HWC -> CHW)
        # If the transform pipeline didn't convert to tensor, do it here.
        if not torch.is_tensor(image_padded):
            image_padded = torch.from_numpy(image_padded).permute(2, 0, 1).float()

        if mask_padded is not None and not torch.is_tensor(mask_padded):
            mask_padded = torch.from_numpy(mask_padded).permute(2, 0, 1).float()

        # Return
        if self.mode != "test":
            return image_padded, mask_padded, self.ids[idx]
        else:
            return image_padded, self.ids[idx]
