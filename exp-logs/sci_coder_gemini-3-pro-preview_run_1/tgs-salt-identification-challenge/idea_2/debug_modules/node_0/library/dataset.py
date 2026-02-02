import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import set_seed

# Constants
ORIG_SIZE = 101
TARGET_SIZE = 128
DEPTH_MIN = 0.0
DEPTH_MAX = 1000.0


def load_and_cache_data(metadata_path, cache_dir, dataset_type, load_cached_data=True):
    """
    Loads data from metadata CSV and images, caching the result as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_dir (str): Directory to store/load cached .npy files.
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    p_images = os.path.join(cache_dir, f"{dataset_type}_images.npy")
    p_masks = os.path.join(cache_dir, f"{dataset_type}_masks.npy")
    p_depths = os.path.join(cache_dir, f"{dataset_type}_depths.npy")
    p_ids = os.path.join(cache_dir, f"{dataset_type}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        # We need images, depths, ids. Masks are optional for test.
        has_basic = (
            os.path.exists(p_images)
            and os.path.exists(p_depths)
            and os.path.exists(p_ids)
        )
        has_masks = os.path.exists(p_masks)

        if has_basic and (dataset_type == "test" or has_masks):
            print(f"Loading {dataset_type} data from cache at {cache_dir}...")
            images = np.load(p_images)
            depths = np.load(p_depths)
            ids = np.load(p_ids, allow_pickle=True)
            masks = np.load(p_masks) if dataset_type != "test" else None
            return images, masks, depths, ids

    # 2. Process from scratch
    print(
        f"Processing {dataset_type} data from scratch (Cache miss or force reload)..."
    )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []

    input_dir = "./input"

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(input_dir, row["image_path"])
        # Read as grayscale to get (H, W)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        images_list.append(img)
        depths_list.append(row["z"])
        ids_list.append(row["id"])

        # Load Mask (if not test)
        if dataset_type != "test":
            mask_path = os.path.join(input_dir, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            # Binarize mask (0 or 255 -> 0 or 1)
            mask = (mask > 127).astype(np.uint8)
            masks_list.append(mask)

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    depths = np.array(depths_list, dtype=np.float32)
    ids = np.array(ids_list)

    if dataset_type != "test":
        masks = np.array(masks_list, dtype=np.uint8)
    else:
        masks = None

    # 3. Save to cache
    print(f"Saving {dataset_type} data to cache at {cache_dir}...")
    np.save(p_images, images)
    np.save(p_depths, depths)
    np.save(p_ids, ids)
    if masks is not None:
        np.save(p_masks, masks)

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(self, images, depths, masks=None, ids=None, is_train=False):
        """
        Args:
            images (np.array): Array of images (N, 101, 101).
            depths (np.array): Array of depths (N,).
            masks (np.array): Array of masks (N, 101, 101) or None.
            ids (np.array): Array of IDs.
            is_train (bool): Whether to apply training augmentations.
        """
        self.images = images
        self.depths = depths
        self.masks = masks
        self.ids = ids
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Fetch data
        image = self.images[idx]  # (101, 101)
        depth = self.depths[idx]  # scalar

        if self.masks is not None:
            mask = self.masks[idx]  # (101, 101)
        else:
            # Dummy mask for test set
            mask = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)

        # Augmentation: Random Horizontal Flip
        if self.is_train and np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        # Preprocessing: Reflection Padding
        # Target: 128x128. Source: 101x101. Diff: 27.
        # Top: 13, Bottom: 14, Left: 13, Right: 14
        pad_h = TARGET_SIZE - ORIG_SIZE
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top

        pad_w = TARGET_SIZE - ORIG_SIZE
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        mask = cv2.copyMakeBorder(
            mask, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

        # Normalization
        # Image: 0-255 -> 0.0-1.0
        image = image.astype(np.float32) / 255.0

        # Depth: Normalize and create a dense channel
        d_norm = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        depth_channel = np.full((TARGET_SIZE, TARGET_SIZE), d_norm, dtype=np.float32)

        # Concatenate Image and Depth
        # Image is (H, W). Stack to make (2, H, W)
        combined_input = np.stack(
            [image, depth_channel], axis=0
        )  # Shape: (2, 128, 128)

        # Convert to Tensors
        input_tensor = torch.tensor(combined_input, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(
            0
        )  # Shape: (1, 128, 128)

        return input_tensor, mask_tensor


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True, debug=False):
    """
    Creates training and validation DataLoaders.
    """
    cache_dir = "./working/idea_2"

    # Load Train Data
    train_imgs, train_masks, train_depths, train_ids = load_and_cache_data(
        "./metadata/train.csv", cache_dir, "train", load_cached_data
    )

    # Load Val Data
    val_imgs, val_masks, val_depths, val_ids = load_and_cache_data(
        "./metadata/val.csv", cache_dir, "val", load_cached_data
    )

    # Debug Subsetting
    if debug:
        print("Debug mode: Truncating datasets...")
        train_imgs, train_masks, train_depths, train_ids = (
            train_imgs[:100],
            train_masks[:100],
            train_depths[:100],
            train_ids[:100],
        )
        val_imgs, val_masks, val_depths, val_ids = (
            val_imgs[:50],
            val_masks[:50],
            val_depths[:50],
            val_ids[:50],
        )

    # Instantiate Datasets
    train_dataset = SaltDataset(
        train_imgs, train_depths, train_masks, train_ids, is_train=True
    )
    val_dataset = SaltDataset(val_imgs, val_depths, val_masks, val_ids, is_train=False)

    # Create Loaders
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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Creates test DataLoader.
    """
    cache_dir = "./working/idea_2"

    # Load Test Data
    test_imgs, _, test_depths, test_ids = load_and_cache_data(
        "./metadata/test.csv", cache_dir, "test", load_cached_data
    )

    # Instantiate Dataset
    test_dataset = SaltDataset(test_imgs, test_depths, None, test_ids, is_train=False)

    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
