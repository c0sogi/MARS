import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import pad_image, rle_decode

# Constants
CACHE_DIR = "./working/idea_15/"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Since we convert to 1-channel grayscale, we average the ImageNet stats
# or use 0.5/0.5 if we assume simple scaling.
# However, standard practice for 1-channel transfer learning is to keep
# the mean/std that matches the pre-trained weights summation logic
# or simply use 0.45 (avg of imagenet mean) and 0.225.
# Let's use the average of ImageNet stats for 1-channel normalization.
GRAY_MEAN = [np.mean(IMAGENET_MEAN)]
GRAY_STD = [np.mean(IMAGENET_STD)]


def load_and_cache_data(
    csv_path, cache_prefix, load_cached_data=True, input_root="./input"
):
    """
    Loads data from CSV, processing images/masks and caching them as .npy files.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    p_images = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    p_masks = os.path.join(CACHE_DIR, f"{cache_prefix}_masks.npy")
    p_depths = os.path.join(CACHE_DIR, f"{cache_prefix}_depths.npy")
    p_ids = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(p_images)
            and os.path.exists(p_depths)
            and os.path.exists(p_ids)
        ):
            # Check mask existence only if it's expected (train/val)
            # We determine if masks are needed by checking if the CSV has 'rle_mask'
            # But here we just check file existence.
            try:
                images = np.load(p_images)
                depths = np.load(p_depths)
                ids = np.load(p_ids)
                masks = np.load(p_masks) if os.path.exists(p_masks) else None
                print(f"Loaded {cache_prefix} data from cache.")
                return images, masks, depths, ids
            except Exception as e:
                print(f"Cache loading failed: {e}. Recomputing...")
        else:
            print(f"Cache miss for {cache_prefix}. Computing...")

    # 2. Compute from scratch
    df = pd.read_csv(csv_path)

    # Lists to collect data
    img_list = []
    mask_list = []
    depth_list = []
    id_list = []

    has_masks = "rle_mask" in df.columns

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(input_root, row["image_path"])
        # Load as grayscale (1 channel)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Pad Image
        img_padded = pad_image(img, target_size=(IMG_SIZE_TARGET, IMG_SIZE_TARGET))
        img_list.append(img_padded)

        # Load Mask if present
        if has_masks:
            rle = row["rle_mask"] if pd.notna(row["rle_mask"]) else ""
            mask = rle_decode(rle, shape=(IMG_SIZE_ORIG, IMG_SIZE_ORIG))
            mask_padded = pad_image(
                mask, target_size=(IMG_SIZE_TARGET, IMG_SIZE_TARGET)
            )
            mask_list.append(mask_padded)

        # Depth
        depth_list.append(row["z"])

        # ID
        id_list.append(str(row["id"]))

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)  # (N, 128, 128)
    depths = np.array(depth_list, dtype=np.float32)
    ids = np.array(id_list)

    if has_masks:
        masks = np.array(mask_list, dtype=np.uint8)  # (N, 128, 128)
    else:
        masks = None

    # 3. Save to cache
    np.save(p_images, images)
    np.save(p_depths, depths)
    np.save(p_ids, ids)
    if masks is not None:
        np.save(p_masks, masks)

    print(f"Processed and cached {len(images)} samples for {cache_prefix}.")

    return images, masks, depths, ids


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # Non-Rigid
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2
                ),
                # Rigid
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.HorizontalFlip(p=0.5),
                # Normalize and Convert
                A.Normalize(mean=GRAY_MEAN, std=GRAY_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose([A.Normalize(mean=GRAY_MEAN, std=GRAY_STD), ToTensorV2()])


class SaltDataset(Dataset):
    def __init__(
        self, images, masks, depths, ids, depth_stats, mode="train", transform=None
    ):
        """
        Args:
            images: (N, H, W) uint8 array
            masks: (N, H, W) uint8 array or None
            depths: (N,) float32 array
            ids: (N,) string array
            depth_stats: tuple (mean, std) for depth normalization
            mode: 'train', 'val', 'test'
            transform: albumentations transform
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.depth_mean, self.depth_std = depth_stats
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get data
        image = self.images[idx]  # (128, 128)
        depth_raw = self.depths[idx]
        img_id = self.ids[idx]

        # Handle Mask
        if self.masks is not None:
            mask = self.masks[idx]  # (128, 128)
        else:
            # Create dummy mask for test set
            mask = np.zeros_like(image)

        # Augmentations
        if self.transform:
            # Albumentations expects HWC or HW
            # For grayscale, passing HW is fine, it treats as single channel
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            # Fallback if no transform provided
            image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        # Mask tensor needs to be float for BCE/Lovasz usually, or long?
        # Losses usually expect float for BCE, but Lovasz might handle either.
        # Let's return float for mask to be safe with BCEWithLogits
        mask_tensor = mask_tensor.float().unsqueeze(0)  # (1, 128, 128)

        # Depth Processing
        # 1. Normalize
        if self.depth_std > 0:
            depth_norm = (depth_raw - self.depth_mean) / self.depth_std
        else:
            depth_norm = depth_raw - self.depth_mean

        # 2. Bernoulli Depth Masking (Train only)
        # With p=0.5, replace depth with 0 (mean)
        if self.mode == "train":
            if np.random.rand() < 0.5:
                depth_norm = 0.0

        # For pseudo-labeling stage where we might pass 'test' mode but want fixed depth 0
        # The caller should handle this by passing depths=0 array or we rely on the fact
        # that test depths are unknown so we might pass 0.
        # But here we stick to the logic: Train gets masking, Val/Test gets raw normalized.

        depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

        return image_tensor, mask_tensor, depth_tensor, img_id


def get_loaders(batch_size=32, num_workers=2, load_cached_data=True, debug=False):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    Calculates depth stats from Train set and applies to all.
    """
    # Paths
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"

    # Load Data Arrays
    train_imgs, train_masks, train_depths, train_ids = load_and_cache_data(
        train_csv, "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = load_and_cache_data(
        val_csv, "val", load_cached_data
    )
    test_imgs, _, test_depths, test_ids = load_and_cache_data(
        test_csv, "test", load_cached_data
    )

    # Debug mode: subset data
    if debug:
        subset_size = 100
        train_imgs, train_masks, train_depths, train_ids = (
            train_imgs[:subset_size],
            train_masks[:subset_size],
            train_depths[:subset_size],
            train_ids[:subset_size],
        )
        val_imgs, val_masks, val_depths, val_ids = (
            val_imgs[:subset_size],
            val_masks[:subset_size],
            val_depths[:subset_size],
            val_ids[:subset_size],
        )
        test_imgs, test_depths, test_ids = (
            test_imgs[:subset_size],
            test_depths[:subset_size],
            test_ids[:subset_size],
        )

    # Calculate Depth Stats from Training Set
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)
    depth_stats = (depth_mean, depth_std)

    # Create Datasets
    train_ds = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        depth_stats,
        mode="train",
        transform=get_transforms("train"),
    )

    val_ds = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        depth_stats,
        mode="val",
        transform=get_transforms("val"),
    )

    # For Test set: The strategy says "Input a constant depth of 0 for all test images".
    # This corresponds to the mean depth (0 after normalization).
    # We can achieve this by passing an array of mean values as depths.
    # Since we normalize by subtracting mean, passing `depth_mean` results in 0.
    test_depths_fixed = np.full_like(test_depths, depth_mean)

    test_ds = SaltDataset(
        test_imgs,
        None,
        test_depths_fixed,
        test_ids,
        depth_stats,
        mode="test",
        transform=get_transforms("test"),
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
