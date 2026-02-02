import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import set_seed

# =========================================================================
# Custom Augmentations
# =========================================================================


class CyclicRoll(A.ImageOnlyTransform):
    """
    Applies a cyclic roll along the time axis (width).
    Used for training augmentation to encourage translation invariance.
    """

    def __init__(self, shift_limit=1.0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.shift_limit = shift_limit

    def apply(self, img, **params):
        # img is assumed to be (H, W, C)
        w = img.shape[1]
        # Random shift between -shift_limit and +shift_limit
        shift_fraction = np.random.uniform(-self.shift_limit, self.shift_limit)
        shift = int(w * shift_fraction)
        return np.roll(img, shift, axis=1)

    def get_transform_init_args_names(self):
        return ("shift_limit",)


class FixedCyclicRoll(A.ImageOnlyTransform):
    """
    Applies a deterministic cyclic roll.
    Used for Test-Time Augmentation (TTA).
    """

    def __init__(self, shift_fraction=0.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.shift_fraction = shift_fraction

    def apply(self, img, **params):
        w = img.shape[1]
        shift = int(w * self.shift_fraction)
        return np.roll(img, shift, axis=1)

    def get_transform_init_args_names(self):
        return ("shift_fraction",)


# =========================================================================
# Transforms
# =========================================================================


def get_transforms(phase="train", tta_shift=None):
    """
    Returns the Albumentations composition of transforms.

    Args:
        phase (str): 'train', 'val', or 'test'.
        tta_shift (float, optional): Fraction of width to roll for TTA.
                                     If provided, overrides random rolling in val/test.
    """
    t_list = []

    # 1. Resize to target resolution (Frequency x Time)
    t_list.append(A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH))

    if phase == "train":
        # Random Cyclic Roll (Time-Shift Invariance)
        # We use p=1.0 to force the model to learn from shifted versions constantly,
        # or p=0.5 to see original frames too.
        # Given the "TTA-Enhanced" strategy, we want strong invariance.
        t_list.append(CyclicRoll(shift_limit=0.5, p=0.8))

        # Coarse Dropout (Proxy for SpecAugment)
        # Masking out blocks of time or frequency
        t_list.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=int(Config.IMG_HEIGHT * 0.1),
                max_width=int(Config.IMG_WIDTH * 0.1),
                min_holes=1,
                fill_value=0,
                p=0.5,
            )
        )

        # Mild Mixup is handled in the training loop, not here.

    elif phase == "test" or phase == "val":
        if tta_shift is not None:
            # Deterministic shift for TTA
            t_list.append(FixedCyclicRoll(shift_fraction=tta_shift, p=1.0))

    # 2. Normalize (ImageNet statistics)
    t_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    # 3. Convert to Tensor
    t_list.append(ToTensorV2())

    return A.Compose(t_list)


# =========================================================================
# Dataset Class
# =========================================================================


class BirdDataset(Dataset):
    def __init__(self, images, labels=None, soft_labels=None, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Hard labels (N, NumClasses).
            soft_labels (np.ndarray, optional): Soft targets for distillation (N, NumClasses).
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.images = images
        self.labels = labels
        self.soft_labels = soft_labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as uint8 (0-255)
        img = self.images[idx]

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        result = {"image": img}

        # Add hard labels if present
        if self.labels is not None:
            result["target"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Add soft labels if present (for distillation)
        if self.soft_labels is not None:
            result["soft_target"] = torch.tensor(
                self.soft_labels[idx], dtype=torch.float32
            )

        return result


# =========================================================================
# Data Loading & Caching
# =========================================================================


def load_dataset_data(phase, load_cached_data=True):
    """
    Loads images and labels for the specified phase.
    Uses caching to speed up subsequent runs.

    Args:
        phase (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        images (np.ndarray): Array of images.
        labels (np.ndarray): Array of labels (or None for test).
        ids (np.ndarray): Array of recording IDs.
    """
    cache_img_path = os.path.join(Config.CACHE_DIR, f"images_{phase}.npy")
    cache_lbl_path = os.path.join(Config.CACHE_DIR, f"labels_{phase}.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"ids_{phase}.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_img_path) and os.path.exists(cache_ids_path):
            # For test set, labels might not exist or be placeholders
            if phase == "test" or os.path.exists(cache_lbl_path):
                print(f"Loading {phase} data from cache...")
                images = np.load(cache_img_path)
                ids = np.load(cache_ids_path)
                labels = (
                    np.load(cache_lbl_path) if os.path.exists(cache_lbl_path) else None
                )
                return images, labels, ids

    # 2. Load from source if cache miss
    print(f"Processing {phase} data from source...")

    # Determine CSV path
    if phase == "train":
        csv_path = Config.TRAIN_CSV
    elif phase == "val":
        csv_path = Config.VAL_CSV
    else:
        csv_path = Config.TEST_CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Identify label columns
    label_cols = [c for c in df.columns if c.startswith("species_")]

    images_list = []
    labels_list = []
    ids_list = []

    missing_count = 0

    for idx, row in df.iterrows():
        rec_id = row["rec_id"]

        # Get correct path for filtered spectrogram
        # Metadata has 'file_path_spec' pointing to 'spectrograms'
        # Config helper redirects to 'filtered_spectrograms'
        img_path = Config.get_spectrogram_path(row["file_path_spec"])

        if not os.path.exists(img_path):
            missing_count += 1
            continue

        # Load Image
        # cv2.imread loads as BGR. We use it as is, treating channels as just features.
        # Pseudo-RGB: The BMPs are likely grayscale or single channel.
        # If single channel, we convert to BGR (3 channels) to match ImageNet weights.
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            missing_count += 1
            continue

        # img is now H, W, 3 (BGR)
        # We keep it as BGR for consistency with cv2, Albumentations handles it.
        # We do NOT resize here to save space, we resize in transforms.
        # However, to stack in numpy array, they must be same size?
        # The EDA showed all images are 1246x256. So we can stack.

        images_list.append(img)
        ids_list.append(rec_id)

        # Get Labels
        if phase != "test":
            lbls = row[label_cols].values.astype(np.float32)
            labels_list.append(lbls)
        else:
            # For test, we can store placeholders or nothing
            labels_list.append(np.zeros(len(label_cols), dtype=np.float32))

    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing and skipped.")

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    ids = np.array(ids_list, dtype=np.int32)

    if labels_list:
        labels = np.array(labels_list, dtype=np.float32)
    else:
        labels = None

    # 3. Save to cache
    print(f"Caching {phase} data to {Config.CACHE_DIR}...")
    np.save(cache_img_path, images)
    np.save(cache_ids_path, ids)
    if labels is not None:
        np.save(cache_lbl_path, labels)

    return images, labels, ids
