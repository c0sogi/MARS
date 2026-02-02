import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def process_grouped_metadata(csv_path, cache_dir, load_cached_data=True):
    """
    Groups metadata by image_id to support Unique Content Sampling.
    Caches the result as a parquet file to speed up initialization.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_dir (str): Directory to store the cached parquet file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: Grouped dataframe with columns [image_id, Cover, JMiPOD, JUNIWARD, UERD].
    """
    filename = os.path.basename(csv_path).replace(".csv", "_grouped.parquet")
    cache_path = os.path.join(cache_dir, filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Warning: Failed to load cache from {cache_path} ({e}). Recomputing..."
            )

    # 2. Compute from scratch
    os.makedirs(cache_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Validate required columns
    if (
        "algo" not in df.columns
        or "file_path" not in df.columns
        or "image_id" not in df.columns
    ):
        raise ValueError(f"CSV {csv_path} missing required columns for grouping.")

    # Pivot the table: One row per image_id, columns for each algo's file path
    # We assume the dataset is perfectly paired (1 Cover + 3 Stego per ID)
    grouped = df.pivot(
        index="image_id", columns="algo", values="file_path"
    ).reset_index()

    # Verify all algorithms are present
    required_cols = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    for col in required_cols:
        if col not in grouped.columns:
            raise ValueError(f"Grouped metadata missing algorithm column: {col}")

    # 3. Save to cache
    grouped.to_parquet(cache_path, index=False)

    return grouped


class StegoDataset(Dataset):
    def __init__(
        self, csv_path, root_dir, mode="train", transform=None, load_cached_data=True
    ):
        """
        Dataset for Steganography Detection.

        Args:
            csv_path (str): Path to metadata CSV.
            root_dir (str): Root directory containing image files.
            mode (str): 'train' or 'val'.
                        If 'train' and Config.unique_content_sampling is True,
                        uses grouped data (Unique Content Sampling).
                        Otherwise uses flat data.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use cached grouped metadata.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # Determine if we use Unique Content Sampling
        # This is typically only for training to show diverse content per epoch
        self.use_grouping = (mode == "train") and Config.unique_content_sampling

        if self.use_grouping:
            self.data = process_grouped_metadata(
                csv_path, Config.cache_dir, load_cached_data
            )
            self.algos = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
        else:
            self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.use_grouping:
            # Unique Content Sampling:
            # Randomly select one variant (Cover or one of the Stego algos)
            selected_algo = np.random.choice(self.algos)
            file_path = row[selected_algo]

            # Determine label: Cover is 0, Stego algorithms are 1
            label = 0.0 if selected_algo == "Cover" else 1.0
        else:
            # Standard Flat Access (Validation or standard training)
            file_path = row["file_path"]
            label = float(row["label"])

        # Load Image
        full_path = os.path.join(self.root_dir, file_path)
        image = cv2.imread(full_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            res = self.transform(image=image)
            image = res["image"]
        else:
            # Fallback normalization if no transform provided
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)  # HWC -> CHW
            image = torch.from_numpy(image)

        return image, torch.tensor(label, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Dataset for Inference.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        full_path = os.path.join(self.root_dir, file_path)
        image = cv2.imread(full_path)

        if image is None:
            # In test, we shouldn't crash, but for this task assuming validity is fine.
            # Returning a blank image as extreme fallback.
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            res = self.transform(image=image)
            image = res["image"]
        else:
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)

        return image_id, image


def get_transforms(mode="train"):
    """
    Returns the Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # D4 Augmentations: Flips and Rot90
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization to [0, 1]
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
