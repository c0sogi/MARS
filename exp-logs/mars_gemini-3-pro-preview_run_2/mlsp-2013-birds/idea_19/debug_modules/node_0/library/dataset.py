import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed


def get_transforms(mode="train", height=224, width=448):
    """
    Returns the Albumentations transformations for the dataset.

    Args:
        mode (str): 'train', 'val', or 'test'.
        height (int): Target height (Frequency axis).
        width (int): Target width (Time axis).

    Returns:
        A.Compose: Composed transformations.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup beta distribution parameter.
        device (str): Device to perform operations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
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


class BirdDataset(Dataset):
    def __init__(
        self,
        csv_file,
        mode="train",
        load_cached_data=True,
        cache_dir="./working/idea_19",
        height=224,
        width=448,
        fixed_roll=None,
    ):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
            cache_dir (str): Directory to store cached .npy files.
            height (int): Target image height.
            width (int): Target image width.
            fixed_roll (int, optional): If set, applies a fixed time roll shift (for TTA).
        """
        self.csv_file = csv_file
        self.mode = mode
        self.height = height
        self.width = width
        self.fixed_roll = fixed_roll
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load metadata
        self.df = pd.read_csv(csv_file)

        # Identify label columns
        self.label_cols = [c for c in self.df.columns if c.startswith("species_")]

        # Load images and labels (with caching)
        self.images, self.labels = self._load_data(load_cached_data)

        # Setup transforms
        self.transforms = get_transforms(mode=mode, height=height, width=width)

    def _load_data(self, load_cached_data):
        """
        Loads images and labels, using caching mechanism.
        """
        # Define cache paths based on mode and dataset size to avoid collisions
        # Using the CSV filename hash or simple mode name if unique enough.
        # Here we use the basename of the csv_file to distinguish train/val/test.
        csv_name = os.path.splitext(os.path.basename(self.csv_file))[0]
        cache_img_path = os.path.join(self.cache_dir, f"images_{csv_name}.npy")
        cache_lbl_path = os.path.join(self.cache_dir, f"labels_{csv_name}.npy")

        if (
            load_cached_data
            and os.path.exists(cache_img_path)
            and os.path.exists(cache_lbl_path)
        ):
            try:
                images = np.load(cache_img_path)
                labels = np.load(cache_lbl_path)
                return images, labels
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Process from scratch
        images = []
        labels = []

        input_root = "./input"

        for idx, row in self.df.iterrows():
            # Construct path to filtered spectrogram
            # Metadata has "supplemental_data/spectrograms/..."
            # We need "supplemental_data/filtered_spectrograms/..."
            rel_path = row["file_path_spec"]
            rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")
            full_path = os.path.join(input_root, rel_path)

            if not os.path.exists(full_path):
                # Fallback to standard spectrogram if filtered is missing (unlikely based on description)
                full_path = os.path.join(input_root, row["file_path_spec"])

            # Load image
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                # Placeholder black image if load fails
                img = np.zeros((self.height, self.width), dtype=np.uint8)
            else:
                # Resize to target resolution
                # cv2.resize takes (width, height)
                img = cv2.resize(
                    img, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                )

            # Convert to Pseudo-RGB (3 channels)
            if len(img.shape) == 2:
                img = cv2.merge([img, img, img])
            elif img.shape[2] == 1:
                img = cv2.merge([img[:, :, 0], img[:, :, 0], img[:, :, 0]])

            images.append(img)

            # Get labels
            lbl = row[self.label_cols].values.astype(np.float32)
            labels.append(lbl)

        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.float32)

        # Save to cache
        np.save(cache_img_path, images)
        np.save(cache_lbl_path, labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()  # Copy to avoid modifying cached array
        label = self.labels[idx]

        # Apply Time-Rolling Augmentation
        # The image is (H, W, C). Time axis is W (axis 1).
        if self.mode == "train":
            # Random circular shift along time axis
            shift = np.random.randint(0, self.width)
            image = np.roll(image, shift, axis=1)
        elif self.fixed_roll is not None:
            # Fixed shift for TTA
            shift = int(self.fixed_roll * self.width)
            image = np.roll(image, shift, axis=1)

        # Apply Albumentations (Normalize -> Tensor)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, label
