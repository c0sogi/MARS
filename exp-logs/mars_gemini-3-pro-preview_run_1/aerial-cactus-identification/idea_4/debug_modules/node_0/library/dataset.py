import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase: str):
    """
    Returns the data augmentation and normalization pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    # Common normalization using stats from Config
    # Albumentations Normalize expects images in [0, 255] and divides by max_pixel_value
    normalize = A.Normalize(
        mean=Config.NORM_MEAN, std=Config.NORM_STD, max_pixel_value=255.0, p=1.0
    )

    to_tensor = ToTensorV2()

    if phase == "train":
        return A.Compose(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), normalize, to_tensor]
        )
    else:
        # val or test
        return A.Compose([normalize, to_tensor])


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Performs Mixup augmentation on the batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup alpha parameter for Beta distribution.
        device (torch.device or str): Device to perform calculations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels of the first image set.
        y_b (torch.Tensor): Labels of the second image set.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class CactusDataset(Dataset):
    """
    Custom Dataset for loading Cactus images from metadata.
    Handles caching of loaded images to speed up training.
    """

    def __init__(
        self, metadata_path, phase, transform=None, debug=False, load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test' (used for cache naming).
            transform (callable, optional): Transform to apply to images.
            debug (bool): If True, limits dataset size for debugging.
            load_cached_data (bool): If True, attempts to load from cache.
        """
        self.metadata_path = metadata_path
        self.phase = phase
        self.transform = transform
        self.debug = debug
        self.load_cached_data = load_cached_data

        # Load data (images and labels) into memory
        self.images, self.labels = self._load_data()

    def _load_data(self):
        """
        Internal method to load data, handling caching logic.
        """
        # Define cache paths
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        imgs_cache_path = os.path.join(cache_dir, f"cache_{self.phase}_imgs.npy")
        labels_cache_path = os.path.join(cache_dir, f"cache_{self.phase}_labels.npy")

        # 1. Attempt to load from cache
        if (
            self.load_cached_data
            and os.path.exists(imgs_cache_path)
            and os.path.exists(labels_cache_path)
        ):
            try:
                print(f"Loading {self.phase} data from cache...")
                images = np.load(imgs_cache_path)
                labels = np.load(labels_cache_path)

                if self.debug:
                    limit = min(len(images), Config.DEBUG_SAMPLES)
                    return images[:limit], labels[:limit]
                return images, labels
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")

        # 2. Process from source if cache miss or debug mode (without cache)
        print(f"Processing {self.phase} data from source...")
        df = pd.read_csv(self.metadata_path)

        if self.debug:
            df = df.head(Config.DEBUG_SAMPLES)

        num_samples = len(df)
        images = np.zeros(
            (num_samples, Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
        )
        labels = np.zeros(num_samples, dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            img = cv2.imread(full_path)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[i] = img
            else:
                # Handle missing images gracefully (keep as zeros)
                pass

            labels[i] = row["has_cactus"]

        # 3. Save to cache ONLY if not debugging
        if not self.debug:
            print(f"Saving {self.phase} data to cache...")
            np.save(imgs_cache_path, images)
            np.save(labels_cache_path, labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Albumentations expects 'image' kwarg
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return label as float tensor for BCEWithLogitsLoss
        return image, torch.tensor(label, dtype=torch.float32)
