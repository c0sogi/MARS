import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import check_tensor_sanitation


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # CoarseDropout acts as a proxy for SpecAugment (Time/Freq masking) on images
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_HEIGHT * 0.1),
                    max_width=int(Config.IMG_WIDTH * 0.1),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()])


def load_and_process_image(rel_path):
    """
    Loads a spectrogram image corresponding to the wav file path,
    resizes it, and applies channel replication.

    Args:
        rel_path (str): Relative path to the wav file (from metadata).

    Returns:
        np.ndarray: Processed image of shape (H, W, 3).
    """
    # Map wav path to spectrogram path
    # Input: essential_data/src_wavs/PC10_... .wav
    # Target: supplemental_data/spectrograms/PC10_... .bmp
    basename = os.path.basename(rel_path)
    bmp_name = os.path.splitext(basename)[0] + ".bmp"
    spectrogram_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)

    if not os.path.exists(spectrogram_path):
        raise FileNotFoundError(f"Spectrogram not found: {spectrogram_path}")

    # Load as Grayscale
    img = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {spectrogram_path}")

    # Densified Global Resize: 256 (H) x 512 (W)
    # cv2.resize expects (width, height)
    img = cv2.resize(
        img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
    )

    # Channel Replication: Gray -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def get_data(metadata_df, load_cached_data=False, cache_prefix="train"):
    """
    Loads dataset images and labels, implementing a strict caching mechanism.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing file paths and labels.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_prefix (str): Prefix for cache filenames (e.g., 'train', 'val').

    Returns:
        tuple: (images, labels, ids) as numpy arrays.
    """
    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(id_cache_path)
        ):
            try:
                images = np.load(img_cache_path)
                labels = np.load(lbl_cache_path)
                ids = np.load(id_cache_path)
                return images, labels, ids
            except Exception:
                # If load fails, proceed to re-compute
                pass

    # 2. Compute from scratch
    images = []
    labels = []
    ids = []

    # Identify label columns
    # Explicitly select columns based on configuration to avoid artifacts (Cite debug_lesson_1)
    expected_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    label_cols = [c for c in expected_cols if c in metadata_df.columns]

    for _, row in metadata_df.iterrows():
        # Process Image
        try:
            img = load_and_process_image(row["file_path"])
            images.append(img)
            ids.append(row["rec_id"])

            # Process Label
            if label_cols:
                lbl = row[label_cols].values.astype(np.float32)
            else:
                # Fallback for test set if columns missing (though metadata usually has them as 0)
                lbl = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            labels.append(lbl)

        except Exception as e:
            # Skip corrupted samples in a robust pipeline, or raise error
            # Here we raise to ensure data integrity
            raise RuntimeError(f"Error processing {row['rec_id']}: {e}")

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(lbl_cache_path, labels)
    np.save(id_cache_path, ids)

    return images, labels, ids


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Supports standard labeled data and pseudo-labeled data for distillation.
    """

    def __init__(self, images, labels, ids, transform=None, pseudo_labels=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray): Array of ground truth labels (N, NumClasses).
            ids (np.ndarray): Array of recording IDs.
            transform (A.Compose, optional): Albumentations transforms.
            pseudo_labels (np.ndarray, optional): Array of soft targets (N, NumClasses).
                                                  If provided, these override ground truth labels.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.pseudo_labels = pseudo_labels

        # Validation
        if self.pseudo_labels is not None:
            if len(self.pseudo_labels) != len(self.images):
                raise ValueError("Pseudo-labels length must match images length.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        rec_id = self.ids[idx]

        # Determine Target (Priority: Pseudo-labels > Ground Truth)
        if self.pseudo_labels is not None:
            target = self.pseudo_labels[idx]
        else:
            target = self.labels[idx]

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback manual normalization if no transform provided
            image = image.astype(np.float32) / 255.0
            image = (image - np.array(Config.MEAN)) / np.array(Config.STD)
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = torch.tensor(image, dtype=torch.float32)

        # Ensure target is tensor
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)

        # Target Sanitation: Critical for distillation stability
        check_tensor_sanitation(target, name=f"Target_ID_{rec_id}")

        return image, target, rec_id
