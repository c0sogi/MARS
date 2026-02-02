import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles filtered spectrograms, resizing, pseudo-RGB conversion,
    augmentations (Roll, SpecAugment), and soft targets for distillation.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray = None,
        soft_labels: np.ndarray = None,
        rec_ids: np.ndarray = None,
        mode: str = "train",
        tta_shift: int = 0,  # 0 to 3 for TTA (0%, 25%, 50%, 75%)
    ):
        """
        Args:
            images (np.ndarray): Pre-loaded images (N, H, W, 3) in uint8.
            labels (np.ndarray): Ground truth labels (N, num_classes).
            soft_labels (np.ndarray): Soft targets for distillation (N, num_classes).
            rec_ids (np.ndarray): Recording IDs.
            mode (str): 'train', 'val', or 'test'.
            tta_shift (int): Index for deterministic TTA shift (0-3).
                             0 = No shift, 1 = 25%, 2 = 50%, 3 = 75%.
                             Only applied if mode != 'train'.
        """
        self.images = images
        self.labels = labels
        self.soft_labels = soft_labels
        self.rec_ids = rec_ids
        self.mode = mode
        self.tta_shift = tta_shift

        # ImageNet Normalization Constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (H, W, 3) uint8
        img = self.images[idx].copy()

        # --- Augmentations ---
        if self.mode == "train":
            # 1. Cyclic Time-Roll
            # Roll along width (axis 1).
            shift = np.random.randint(0, img.shape[1])
            img = np.roll(img, shift, axis=1)

            # 2. SpecAugment (Simplified)
            # Masking blocks of frequency (height) or time (width)
            # Probability of applying mask
            if np.random.rand() < 0.5:
                # Time Masking
                t_width = np.random.randint(10, 60)
                t_start = np.random.randint(0, img.shape[1] - t_width)
                img[:, t_start : t_start + t_width, :] = 0

            if np.random.rand() < 0.5:
                # Frequency Masking
                f_width = np.random.randint(5, 30)
                f_start = np.random.randint(0, img.shape[0] - f_width)
                img[f_start : f_start + f_width, :, :] = 0

        elif self.tta_shift > 0:
            # Deterministic TTA Shift
            # 0: 0%, 1: 25%, 2: 50%, 3: 75%
            width = img.shape[1]
            shift_amount = int(width * (self.tta_shift * 0.25))
            img = np.roll(img, shift_amount, axis=1)

        # --- Normalization ---
        # Convert to float 0-1
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        img = (img - self.mean) / self.std

        # --- ToTensor ---
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img)

        # Prepare return dict
        sample = {
            "image": img_tensor,
            "rec_id": self.rec_ids[idx] if self.rec_ids is not None else -1,
        }

        if self.labels is not None:
            sample["target"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.soft_labels is not None:
            sample["soft_target"] = torch.tensor(
                self.soft_labels[idx], dtype=torch.float32
            )

        return sample


def process_images(df: pd.DataFrame) -> np.ndarray:
    """
    Loads, resizes, and converts images to Pseudo-RGB.
    Returns a numpy array of shape (N, H, W, 3) uint8.
    """
    images = []
    for _, row in df.iterrows():
        # Construct full path
        rel_path = row["file_path_spec"]
        # The metadata path is relative to input/, but Config.IMAGE_DIR points to filtered_spectrograms
        # We need to be careful. Metadata says 'supplemental_data/spectrograms/...'
        # But Idea 30 uses 'filtered_spectrograms'.
        # We should use the filename from metadata and construct path to filtered_spectrograms.

        filename = os.path.basename(rel_path)
        full_path = os.path.join(Config.IMAGE_DIR, filename)

        if not os.path.exists(full_path):
            # Fallback to standard spectrogram if filtered is missing (unlikely)
            # Or raise error. Given the dataset, we assume existence.
            # For safety, create a blank image
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
        else:
            # Load BMP (Grayscale)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                # Resize
                # cv2.resize takes (width, height)
                img = cv2.resize(
                    img,
                    (Config.IMG_WIDTH, Config.IMG_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Pseudo-RGB (Stack 3 channels)
                img = np.stack([img, img, img], axis=-1)

        images.append(img)

    return np.array(images, dtype=np.uint8)


def get_processed_data(split: str, load_cached_data: bool = True):
    """
    Retrieves the processed data (images, labels, rec_ids) for a given split.
    Implements caching using .npy files in Config.CACHE_DIR.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, rec_ids)
            images: np.ndarray (N, H, W, 3)
            labels: np.ndarray (N, Num_Classes) or None for test
            rec_ids: np.ndarray (N,)
    """
    cache_file_imgs = os.path.join(Config.CACHE_DIR, f"images_{split}.npy")
    cache_file_lbls = os.path.join(Config.CACHE_DIR, f"labels_{split}.npy")
    cache_file_ids = os.path.join(Config.CACHE_DIR, f"ids_{split}.npy")

    # Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_file_imgs) and os.path.exists(cache_file_ids):
            # For test split, labels might not exist or be dummy
            if split == "test" or os.path.exists(cache_file_lbls):
                print(f"Loading {split} data from cache...")
                images = np.load(cache_file_imgs)
                rec_ids = np.load(cache_file_ids)
                labels = (
                    np.load(cache_file_lbls)
                    if os.path.exists(cache_file_lbls)
                    else None
                )
                return images, labels, rec_ids

    print(f"Processing {split} data from scratch...")

    # Load Metadata
    if split == "train":
        csv_path = Config.TRAIN_CSV
    elif split == "val":
        csv_path = Config.VAL_CSV
    else:
        csv_path = Config.TEST_CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Process Images
    images = process_images(df)

    # Process IDs
    rec_ids = df["rec_id"].values

    # Process Labels
    label_cols = [c for c in df.columns if c.startswith("species_")]
    if label_cols:
        labels = df[label_cols].values.astype(np.float32)
    else:
        labels = None

    # Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_file_imgs, images)
    np.save(cache_file_ids, rec_ids)
    if labels is not None:
        np.save(cache_file_lbls, labels)

    return images, labels, rec_ids
