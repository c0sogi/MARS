import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config
from library.utils import pad_image_128


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_ALPHA_AFFINE,
                    p=Config.AUG_PROB,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_PROB,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )
    else:
        return A.Compose([])


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles padding, augmentation, and normalization.
    """

    def __init__(
        self, images, masks=None, depths=None, ids=None, transform=None, mode="train"
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W) or (N, H, W, C).
            masks (np.ndarray, optional): Array of masks (N, H, W). Binary or Soft.
            depths (np.ndarray, optional): Array of scaled depths (N,).
            ids (list, optional): List of image IDs.
            transform (A.Compose, optional): Albumentations pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.mode = mode

        # Calculate 1-channel normalization stats from Config (RGB)
        self.norm_mean = np.mean(Config.NORM_MEAN)
        self.norm_std = np.mean(Config.NORM_STD)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load Image
        img = self.images[idx]  # (101, 101)

        # 2. Pad to 128x128 (Reflection)
        img = pad_image_128(img)

        # 3. Handle Mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            mask = pad_image_128(mask)  # (128, 128)

        # 4. Augmentations
        if self.transform:
            # Albumentations requires HWC
            if img.ndim == 2:
                img = img[:, :, None]

            if mask is not None:
                if mask.ndim == 2:
                    mask = mask[:, :, None]

                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        else:
            # Ensure HWC for consistency even without transforms
            if img.ndim == 2:
                img = img[:, :, None]
            if mask is not None and mask.ndim == 2:
                mask = mask[:, :, None]

        # 5. Normalize
        # Convert to float [0, 1]
        img = img.astype(np.float32) / 255.0
        # Standardize
        img = (img - self.norm_mean) / self.norm_std

        # 6. To Tensor (HWC -> CHW)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()

        data = {"image": img_tensor}

        if self.ids is not None:
            data["id"] = self.ids[idx]

        if mask is not None:
            # Mask to Tensor (HWC -> CHW)
            mask = mask.transpose(2, 0, 1)
            data["mask"] = torch.from_numpy(mask).float()

        if self.depths is not None:
            d = self.depths[idx]
            data["depth"] = torch.tensor([d], dtype=torch.float32)

        return data


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches dataset arrays.

    Args:
        load_cached_data (bool): If True, attempts to load from ./working/idea_35/

    Returns:
        dict: Contains 'train', 'val', 'test' dictionaries with keys:
              'images', 'masks', 'depths', 'ids'.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    modes = ["train", "val", "test"]
    data_containers = {}

    # Check if all cache files exist
    cache_files = {}
    all_cached = True
    for mode in modes:
        cache_files[mode] = {
            "images": os.path.join(cache_dir, f"{mode}_images.npy"),
            "masks": os.path.join(cache_dir, f"{mode}_masks.npy"),
            "depths": os.path.join(cache_dir, f"{mode}_depths.npy"),
            "ids": os.path.join(cache_dir, f"{mode}_ids.npy"),
        }
        # Test set doesn't strictly need masks in cache unless we have pseudo labels,
        # but here we follow standard structure. Test masks will be placeholders or None.
        for k, v in cache_files[mode].items():
            if k == "masks" and mode == "test":
                continue
            if not os.path.exists(v):
                all_cached = False
                break

    # 1. Load from Cache
    if load_cached_data and all_cached:
        print("Loading data from cache...")
        for mode in modes:
            images = np.load(cache_files[mode]["images"])
            depths = np.load(cache_files[mode]["depths"])
            ids = np.load(cache_files[mode]["ids"], allow_pickle=True)

            masks = None
            if mode != "test":
                masks = np.load(cache_files[mode]["masks"])

            data_containers[mode] = {
                "images": images,
                "masks": masks,
                "depths": depths,
                "ids": ids,
            }
        return data_containers

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Metadata
    meta = {
        "train": pd.read_csv(Config.TRAIN_METADATA),
        "val": pd.read_csv(Config.VAL_METADATA),
        "test": pd.read_csv(Config.TEST_METADATA),
    }

    # Compute Depth Statistics (Standard Scaling) using ONLY Train set
    train_depths = meta["train"]["z"].values.astype(np.float32)
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)

    # Save depth stats for inference reference (optional, but good practice)
    # We embed them into the processed arrays.

    for mode in modes:
        df = meta[mode]
        count = len(df)

        # Pre-allocate arrays
        # Images: (N, 101, 101) uint8
        img_arr = np.zeros(
            (count, Config.IMG_ORIG_SIZE, Config.IMG_ORIG_SIZE), dtype=np.uint8
        )
        # Masks: (N, 101, 101) uint8 (Binary) - Only for train/val
        mask_arr = (
            np.zeros(
                (count, Config.IMG_ORIG_SIZE, Config.IMG_ORIG_SIZE), dtype=np.uint8
            )
            if mode != "test"
            else None
        )
        # Depths: (N,) float32
        depth_arr = np.zeros((count,), dtype=np.float32)
        # IDs: List
        ids_list = []

        print(f"Loading {mode} images...")
        for idx, row in df.iterrows():
            # Load Image
            img_path = os.path.join(Config.INPUT_ROOT, row["image_path"])
            # Read as grayscale (IMREAD_GRAYSCALE)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img_arr[idx] = img

            # Load Mask (if available)
            if mode != "test":
                mask_path = os.path.join(Config.INPUT_ROOT, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # Some masks might be empty/missing? Metadata guarantees existence.
                    # If read fails, assume empty.
                    mask = np.zeros_like(img)
                # Ensure binary 0/1
                mask = (mask > 127).astype(np.uint8)
                mask_arr[idx] = mask

            # Process Depth
            z = row["z"]
            z_norm = (z - depth_mean) / depth_std
            depth_arr[idx] = z_norm

            ids_list.append(row["id"])

        # Convert IDs to numpy for caching
        ids_arr = np.array(ids_list)

        # Save to Cache
        np.save(cache_files[mode]["images"], img_arr)
        np.save(cache_files[mode]["depths"], depth_arr)
        np.save(cache_files[mode]["ids"], ids_arr)
        if mask_arr is not None:
            np.save(cache_files[mode]["masks"], mask_arr)

        data_containers[mode] = {
            "images": img_arr,
            "masks": mask_arr,
            "depths": depth_arr,
            "ids": ids_arr,
        }

    print("Data processing complete and cached.")
    return data_containers
