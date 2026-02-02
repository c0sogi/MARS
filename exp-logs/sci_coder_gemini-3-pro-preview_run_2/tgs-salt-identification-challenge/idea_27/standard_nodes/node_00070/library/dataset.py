import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_27"
IMG_ORIG_SIZE = 101
IMG_TARGET_SIZE = 128


def get_depth_stats(root_dir=INPUT_DIR):
    """
    Calculates global depth mean and std from depths.csv.
    """
    depths_path = os.path.join(root_dir, "depths.csv")
    if os.path.exists(depths_path):
        df = pd.read_csv(depths_path)
        return df["z"].mean(), df["z"].std()
    # Fallback defaults if file missing
    return 0.0, 1.0


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes RLE mask string to numpy array.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    # Mean/Std for Grayscale (approx ImageNet average)
    # R=0.485, G=0.456, B=0.406 -> Avg ~ 0.449
    # Std ~ 0.226
    mean = [0.449]
    std = [0.226]

    if mode == "train":
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_TARGET_SIZE,
                    min_width=IMG_TARGET_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.2
                ),
                # Elastic Transform parameters: alpha=120, sigma=6, alpha_affine=3.6
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test / Pseudo
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_TARGET_SIZE,
                    min_width=IMG_TARGET_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    def __init__(
        self,
        mode="train",
        root_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        transform=None,
        cache_dir=CACHE_DIR,
        load_cached_data=True,
    ):
        """
        Args:
            mode (str): 'train', 'val', 'test', or 'pseudo'.
            root_dir (str): Path to input directory.
            metadata_dir (str): Path to metadata directory.
            transform (A.Compose): Albumentations transforms.
            cache_dir (str): Path to store/load cached npy files.
            load_cached_data (bool): Whether to use caching.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data

        # Load Global Depth Stats
        self.depth_mean, self.depth_std = get_depth_stats(root_dir)

        # Determine Metadata File
        if mode == "train":
            csv_file = "train.csv"
        elif mode == "val":
            csv_file = "val.csv"
        elif mode == "test":
            csv_file = "test.csv"
        elif mode == "pseudo":
            # Try to find pseudo_train.csv, else fallback to train.csv
            csv_file = "pseudo_train.csv"
            if not os.path.exists(os.path.join(metadata_dir, csv_file)):
                csv_file = "train.csv"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.meta_path = os.path.join(metadata_dir, csv_file)
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.df = pd.read_csv(self.meta_path)

        # Load Data (Images, Masks, Depths)
        self._load_data()

    def _load_data(self):
        """
        Loads data from cache or processes from raw files.
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache paths
        images_cache = os.path.join(self.cache_dir, f"{self.mode}_images.npy")
        masks_cache = os.path.join(self.cache_dir, f"{self.mode}_masks.npy")
        depths_cache = os.path.join(self.cache_dir, f"{self.mode}_depths.npy")
        ids_cache = os.path.join(self.cache_dir, f"{self.mode}_ids.npy")

        # Check if cache exists
        cache_exists = (
            os.path.exists(images_cache)
            and os.path.exists(depths_cache)
            and os.path.exists(ids_cache)
        )

        if self.mode != "test":
            cache_exists = cache_exists and os.path.exists(masks_cache)

        if self.load_cached_data and cache_exists:
            # Load from cache
            self.images = np.load(images_cache)
            self.depths = np.load(depths_cache)
            self.ids = np.load(ids_cache, allow_pickle=True)
            if self.mode != "test":
                self.masks = np.load(masks_cache)
            else:
                self.masks = None
        else:
            # Process from scratch
            images_list = []
            masks_list = []
            depths_list = []
            ids_list = []

            for idx, row in self.df.iterrows():
                # Read Image
                # Metadata contains relative path: 'train/images/xxxx.png'
                img_path = os.path.join(self.root_dir, row["image_path"])
                # Read as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    # Safety fallback
                    img = np.zeros((IMG_ORIG_SIZE, IMG_ORIG_SIZE), dtype=np.uint8)

                images_list.append(img)
                # Use depth from metadata (merged from depths.csv)
                depths_list.append(row["z"])
                ids_list.append(row["id"])

                if self.mode != "test":
                    # Decode Mask
                    if "rle_mask" in row and not pd.isna(row["rle_mask"]):
                        mask = rle_decode(row["rle_mask"])
                    else:
                        mask = np.zeros((IMG_ORIG_SIZE, IMG_ORIG_SIZE), dtype=np.uint8)
                    masks_list.append(mask)

            # Convert to arrays
            self.images = np.array(images_list, dtype=np.uint8)
            self.depths = np.array(depths_list, dtype=np.float32)
            self.ids = np.array(ids_list)

            # Save to cache
            np.save(images_cache, self.images)
            np.save(depths_cache, self.depths)
            np.save(ids_cache, self.ids)

            if self.mode != "test":
                self.masks = np.array(masks_list, dtype=np.uint8)
                np.save(masks_cache, self.masks)
            else:
                self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get data
        image = self.images[idx]  # (H, W)
        depth = self.depths[idx]
        id_ = self.ids[idx]

        # Normalize depth
        depth = (depth - self.depth_mean) / self.depth_std
        depth = torch.tensor([depth], dtype=torch.float32)

        if self.mode != "test":
            mask = self.masks[idx]  # (H, W)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                # Fallback to ToTensor if no transform
                image = torch.from_numpy(image).float().unsqueeze(0)
                mask = torch.from_numpy(mask).float()

            # Ensure mask is float for BCE
            mask = mask.float()

            return image, mask, depth, id_
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = torch.from_numpy(image).float().unsqueeze(0)

            return image, depth, id_
