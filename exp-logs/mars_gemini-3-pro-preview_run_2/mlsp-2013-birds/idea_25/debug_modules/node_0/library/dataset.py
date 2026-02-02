import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class CyclicRoll(A.ImageOnlyTransform):
    """
    Custom Albumentations transform that performs a random cyclic shift
    along the time axis (width).
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # img is expected to be (H, W, C)
        # Time axis is W (axis 1)
        h, w, c = img.shape
        shift = np.random.randint(0, w)
        return np.roll(img, shift, axis=1)

    def get_transform_init_args_names(self):
        return ()


def get_train_transforms():
    """
    Returns the training augmentations:
    - Cyclic Time-Rolling
    - SpecAugment (simulated via CoarseDropout)
    - RandomBrightnessContrast
    - Normalization & ToTensor
    """
    return A.Compose(
        [
            # Cyclic Time-Rolling: Random circular shift along time axis
            CyclicRoll(p=1.0),
            # SpecAugment simulation: Masking blocks of time/freq
            A.CoarseDropout(
                max_holes=8,
                max_height=int(Config.IMG_HEIGHT * 0.1),
                max_width=int(Config.IMG_WIDTH * 0.1),
                min_holes=1,
                fill_value=0,
                p=0.5,
            ),
            # Photometric augmentations
            A.RandomBrightnessContrast(p=0.5),
            # Standard Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    """
    Returns the validation/test transforms:
    - Normalization & ToTensor
    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles loading filtered spectrograms, resizing, Pseudo-RGB conversion,
    and caching to disk for speed.
    """

    def __init__(self, csv_path, mode="train", load_cached_data=True, transforms=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Used for cache naming.
            load_cached_data (bool): If True, attempts to load pre-processed data from .npy files.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.csv_path = csv_path
        self.mode = mode
        self.transforms = transforms

        # Define cache paths
        self.cache_dir = Config.WORK_DIR
        self.images_cache_path = os.path.join(self.cache_dir, f"images_{self.mode}.npy")
        self.labels_cache_path = os.path.join(self.cache_dir, f"labels_{self.mode}.npy")
        self.ids_cache_path = os.path.join(self.cache_dir, f"ids_{self.mode}.npy")

        self.images = None
        self.labels = None
        self.rec_ids = None

        # Load data
        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        # 1. Try loading from cache
        if load_cached_data:
            if (
                os.path.exists(self.images_cache_path)
                and os.path.exists(self.labels_cache_path)
                and os.path.exists(self.ids_cache_path)
            ):
                try:
                    self.images = np.load(self.images_cache_path)
                    self.labels = np.load(self.labels_cache_path)
                    self.rec_ids = np.load(self.ids_cache_path)
                    return
                except Exception as e:
                    print(f"Failed to load cache for {self.mode}: {e}. Recomputing...")

        # 2. Compute from scratch if cache missing or failed
        df = pd.read_csv(self.csv_path)

        # Debugging: Limit size if Config.DEBUG is set
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SAMPLES)

        img_list = []
        label_list = []
        id_list = []

        # Identify label columns
        label_cols = [c for c in df.columns if c.startswith("species_")]

        for idx, row in df.iterrows():
            # Get filename from metadata and construct path to FILTERED spectrograms
            # Metadata path example: supplemental_data/spectrograms/PC10_....bmp
            # We want: supplemental_data/filtered_spectrograms/PC10_....bmp
            orig_rel_path = row["file_path_spec"]
            filename = os.path.basename(orig_rel_path)
            full_path = os.path.join(Config.SPECTROGRAM_DIR, filename)

            # Load Image
            if not os.path.exists(full_path):
                # Fallback to unfiltered if filtered is missing (should not happen based on task desc)
                full_path = os.path.join(Config.INPUT_DIR, orig_rel_path)

            # Read as grayscale (BMPs are single channel usually, but cv2 reads as BGR by default)
            # Using IMREAD_GRAYSCALE to be explicit
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # Placeholder for missing files (black image)
                img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)
            else:
                # Resize to target dimensions (Frequency x Time) -> (224, 448)
                # cv2.resize expects (width, height)
                img = cv2.resize(
                    img,
                    (Config.IMG_WIDTH, Config.IMG_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Pseudo-RGB: Stack to 3 channels
            img = np.stack([img, img, img], axis=-1)  # (H, W, 3)

            img_list.append(img)
            id_list.append(row["rec_id"])

            # Process Labels
            labels = row[label_cols].values.astype(np.float32)
            label_list.append(labels)

        self.images = np.array(img_list, dtype=np.uint8)
        self.labels = np.array(label_list, dtype=np.float32)
        self.rec_ids = np.array(id_list, dtype=np.int64)

        # 3. Save to cache
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(self.images_cache_path, self.images)
        np.save(self.labels_cache_path, self.labels)
        np.save(self.ids_cache_path, self.rec_ids)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        rec_id = self.rec_ids[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to basic tensor conversion if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float32), rec_id


class Mixup:
    """
    Implements Mixup augmentation.
    To be used inside the training loop.
    """

    def __init__(self, alpha=Config.MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Applies mixup to a batch.
        Args:
            batch: tuple of (images, labels, ids)
        Returns:
            mixed_images, labels_a, labels_b, lam
        """
        images, labels, _ = batch
        batch_size = images.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam
