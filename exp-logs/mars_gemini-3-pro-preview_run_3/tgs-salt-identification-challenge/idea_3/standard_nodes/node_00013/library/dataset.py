import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.config import Config


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_image(img, target_size=128):
    """
    Pads image from (H, W) to (target_size, target_size) using reflection padding.
    Reflection padding is ideal for seismic data to avoid boundary artifacts.
    """
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w

    if pad_h <= 0 and pad_w <= 0:
        return img

    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Use REFLECT_101 to avoid repeating the edge pixel
    return cv2.copyMakeBorder(
        img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def preprocess_data(df, mode, load_cached_data=True):
    """
    Loads, resizes/pads, and caches data.
    Returns: images (np.array), masks (np.array or None), depths (np.array), ids (list)
    """
    cache_dir = Config.IDEA_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    cache_prefix = f"cached_{mode}"
    if Config.DEBUG:
        cache_prefix += "_debug"

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"{cache_prefix}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"{cache_prefix}_depths.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    has_masks = "mask_path" in df.columns

    # Try loading from cache
    if load_cached_data:
        try:
            if (
                os.path.exists(img_cache_path)
                and os.path.exists(depth_cache_path)
                and os.path.exists(id_cache_path)
            ):
                # Check mask cache existence if needed
                if has_masks and not os.path.exists(mask_cache_path):
                    raise FileNotFoundError("Mask cache missing")

                images = np.load(img_cache_path)
                depths = np.load(depth_cache_path)
                ids = np.load(id_cache_path, allow_pickle=True)
                masks = np.load(mask_cache_path) if has_masks else None

                # Verify shape matches current configuration (N, H, W)
                if (
                    images.ndim != 3
                    or images.shape[1] != Config.IMAGE_SIZE
                    or images.shape[2] != Config.IMAGE_SIZE
                ):
                    raise ValueError(
                        f"Cached images shape {images.shape} does not match Config.IMAGE_SIZE {Config.IMAGE_SIZE}"
                    )

                # Verify length matches current dataframe
                if len(images) == len(df):
                    print(f"Loaded {mode} data from cache.")
                    return images, masks, depths, ids
                else:
                    print(f"Cache size mismatch for {mode}. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {mode} data...")
    images = []
    masks = [] if has_masks else None
    depths = df["z"].values.astype(np.float32)
    ids = df["id"].values

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Pad Image
        img_padded = pad_image(img, Config.IMAGE_SIZE)
        images.append(img_padded)

        # Load Mask if exists
        if has_masks:
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback, though metadata checks should prevent this
                mask = np.zeros_like(img)

            # Pad Mask
            mask_padded = pad_image(mask, Config.IMAGE_SIZE)
            # Ensure binary
            mask_padded = (mask_padded > 127).astype(np.uint8)
            masks.append(mask_padded)

    images = np.array(images, dtype=np.uint8)
    if has_masks:
        masks = np.array(masks, dtype=np.uint8)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)
    if has_masks:
        np.save(mask_cache_path, masks)

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, transforms=None):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W) uint8
        depth = self.depths[idx]  # scalar float

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  # (H, W) uint8

        # Apply Augmentations
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=img)
                img = augmented["image"]

        # Normalize Image to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Create Depth Channel (normalized)
        # Depth range is approx 50-960. Dividing by 1000 keeps it in [0, 1]
        depth_val = depth / 1000.0
        depth_channel = np.full_like(img, depth_val, dtype=np.float32)

        # Fuse Channels: [Image, Image, Depth]
        # We duplicate the grayscale image to the first two channels to utilize
        # ImageNet pretrained weights (which expect 3 channels) and place depth in the 3rd.
        input_tensor = np.stack([img, img, depth_channel], axis=0)
        input_tensor = torch.from_numpy(input_tensor).float()

        if mask is not None:
            # Mask to Tensor (1, H, W)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
            return input_tensor, mask_tensor
        else:
            # For inference, return ID to track predictions
            return input_tensor, self.ids[idx]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        # Cite solution_lesson_node_00008: Avoid aggressive geometric distortions on low-resolution seismic data.
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
    else:
        # No test-time augmentations in the dataset loop (handled in inference loop if TTA is on)
        return A.Compose([])


def get_dataloaders(load_cached_data=True):
    """
    Prepares and returns DataLoaders for Train, Val, and Test sets.
    """
    set_seed(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if Config.DEBUG:
        train_df = train_df.iloc[: Config.MAX_DEBUG_SAMPLES]
        val_df = val_df.iloc[: Config.MAX_DEBUG_SAMPLES]
        test_df = test_df.iloc[: Config.MAX_DEBUG_SAMPLES]

    # Preprocess Data (Load/Cache)
    train_imgs, train_masks, train_depths, train_ids = preprocess_data(
        train_df, "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = preprocess_data(
        val_df, "val", load_cached_data
    )
    test_imgs, test_masks, test_depths, test_ids = preprocess_data(
        test_df, "test", load_cached_data
    )

    # Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transforms=get_transforms("train"),
    )
    val_dataset = SaltDataset(
        val_imgs, val_masks, val_depths, val_ids, transforms=get_transforms("val")
    )
    test_dataset = SaltDataset(
        test_imgs, test_masks, test_depths, test_ids, transforms=get_transforms("test")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
