import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_transforms(split="train"):
    """
    Returns the Albumentations transformations for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if split == "train":
        return A.Compose(
            [
                # Note: Cyclic Time-Rolling is handled in the Dataset __getitem__
                # because it requires numpy.roll which isn't standard in Albumentations
                # Simulate SpecAugment with CoarseDropout
                # masking out rectangular regions in time/frequency
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_HEIGHT * 0.1),
                    max_width=int(Config.IMG_WIDTH * 0.1),
                    min_holes=2,
                    fill_value=0,
                    p=0.5,
                ),
                # Random Brightness/Contrast to handle intensity variations
                A.RandomBrightnessContrast(p=0.5),
                # Normalize using ImageNet mean/std (standard for Pseudo-RGB)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def load_data(csv_path, split, load_cached_data=True):
    """
    Loads dataset from CSV, processing images and caching them as .npy files.

    Args:
        csv_path (str): Path to the metadata CSV file.
        split (str): 'train', 'val', or 'test' (used for cache naming).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, rec_ids)
            images: np.ndarray of shape (N, H, W, 3)
            labels: np.ndarray of shape (N, Num_Classes)
            rec_ids: np.ndarray of shape (N,)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache filenames
    debug_suffix = "_debug" if Config.DEBUG else ""
    img_cache_path = os.path.join(
        Config.WORKING_DIR, f"images_{split}{debug_suffix}.npy"
    )
    lbl_cache_path = os.path.join(
        Config.WORKING_DIR, f"labels_{split}{debug_suffix}.npy"
    )
    ids_cache_path = os.path.join(Config.WORKING_DIR, f"ids_{split}{debug_suffix}.npy")

    # Attempt to load from cache
    if load_cached_data and not Config.DEBUG:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            try:
                images = np.load(img_cache_path)
                labels = np.load(lbl_cache_path)
                rec_ids = np.load(ids_cache_path)
                # print(f"Loaded {split} data from cache.")
                return images, labels, rec_ids
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    images = []
    labels = []
    rec_ids = []

    # Identify label columns
    label_cols = [c for c in df.columns if c.startswith("species_")]

    for idx, row in df.iterrows():
        rec_id = row["rec_id"]

        # Construct path to Filtered Spectrogram
        # Metadata points to standard spectrograms, we map to filtered ones
        fname = os.path.basename(row["file_path_spec"])
        img_path = os.path.join(Config.FILTERED_SPEC_DIR, fname)

        # Load Image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Fallback to standard spectrogram if filtered is missing (safety net)
        if img is None:
            std_path = os.path.join(Config.INPUT_DIR, row["file_path_spec"])
            img = cv2.imread(std_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Last resort: black image
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Resize to Target Dimensions (224 Freq x 448 Time)
        # cv2.resize expects (width, height)
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))

        # Convert to Pseudo-RGB (Replicate channels)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        images.append(img)
        rec_ids.append(rec_id)

        # Parse Labels
        if label_cols:
            labels.append(row[label_cols].values.astype(np.float32))
        else:
            # Should not happen given metadata format, but handle gracefully
            labels.append(np.zeros(Config.NUM_CLASSES, dtype=np.float32))

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32)
    rec_ids = np.array(rec_ids, dtype=np.int64)

    # Save to cache (skip if debugging to avoid polluting cache)
    if not Config.DEBUG:
        np.save(img_cache_path, images)
        np.save(lbl_cache_path, labels)
        np.save(ids_cache_path, rec_ids)
        # print(f"Processed and cached {split} data.")

    return images, labels, rec_ids


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Implements Cyclic Time-Rolling augmentation.
    """

    def __init__(self, images, labels, rec_ids, split="train", tta_shift=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray): Array of labels (N, Num_Classes).
            rec_ids (np.ndarray): Array of recording IDs.
            split (str): 'train', 'val', or 'test'.
            tta_shift (float, optional): Fraction of width to roll for TTA.
                                         If None and split='train', random roll is applied.
        """
        self.images = images
        self.labels = labels
        self.rec_ids = rec_ids
        self.split = split
        self.tta_shift = tta_shift
        self.transforms = get_transforms(split)
        self.img_width = Config.IMG_WIDTH

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx]
        rec_id = self.rec_ids[idx]

        # --- Cyclic Time-Rolling ---
        # Axis 1 is Width (Time) for (H, W, C) image
        if self.tta_shift is not None:
            # Deterministic shift for TTA
            shift = int(self.img_width * self.tta_shift)
            image = np.roll(image, shift, axis=1)
        elif self.split == "train":
            # Random cyclic shift for training
            shift = np.random.randint(0, self.img_width)
            image = np.roll(image, shift, axis=1)

        # Apply Albumentations
        augmented = self.transforms(image=image)
        image_tensor = augmented["image"]

        return image_tensor, torch.tensor(label, dtype=torch.float32), rec_id


class MixupCollate:
    """
    Collate function for Mixup Augmentation.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (image, label, rec_id)

        Returns:
            mixed_images, mixed_labels, rec_ids
        """
        batch_size = len(batch)

        # Unpack batch
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        rec_ids = torch.tensor([item[2] for item in batch])

        # Apply Mixup if alpha > 0
        if self.alpha > 0 and batch_size > 1:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Shuffle batch indices
            index = torch.randperm(batch_size)

            # Mix images and labels
            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]

            return mixed_images, mixed_labels, rec_ids

        return images, labels, rec_ids
