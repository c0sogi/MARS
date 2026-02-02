import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class StegoDataset(Dataset):
    def __init__(self, csv_path, mode="train", transform=None, load_cached_data=True):
        """
        Dataset for Steganalysis.

        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to try loading processed metadata from cache.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Load Data
        if self.mode == "train":
            self._init_train(csv_path, load_cached_data)
        else:
            self._init_inference(csv_path)

    def _init_train(self, csv_path, load_cached_data):
        """
        Initializes training data with balancing strategy and caching.
        """
        cache_name = (
            "train_grouped_debug.parquet" if Config.DEBUG else "train_grouped.parquet"
        )
        cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        loaded = False
        df_grouped = pd.DataFrame()

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df_grouped = pd.read_parquet(cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}")

        # 2. Process if not loaded
        if not loaded:
            df = pd.read_csv(csv_path)

            if Config.DEBUG:
                unique_ids = df["image_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
                df = df[df["image_id"].isin(unique_ids)].copy()

            # Pivot to group variants: index=image_id, columns=algo, values=file_path
            # We assume the metadata is complete (every ID has all 4 variants)
            # Filter columns of interest
            df_subset = df[["image_id", "algo", "file_path"]]
            df_grouped = df_subset.pivot(
                index="image_id", columns="algo", values="file_path"
            )

            # Save to cache
            try:
                df_grouped.to_parquet(cache_path)
            except Exception as e:
                print(f"Failed to save cache: {e}")

        # Convert to dictionary for O(1) access
        # 'index' orientation: {image_id: {'Cover': path, 'JMiPOD': path, ...}, ...}
        self.data_map = df_grouped.to_dict("index")
        self.samples = list(self.data_map.keys())

    def _init_inference(self, csv_path):
        """
        Initializes validation/test data (static list).
        """
        df = pd.read_csv(csv_path)

        if Config.DEBUG:
            df = df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

        # Convert to list of dicts
        self.samples = df.to_dict("records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Dynamic Balancing Strategy
            img_id = self.samples[idx]
            variants = self.data_map[img_id]

            # 50% probability for Cover (0), 50% for Stego (1)
            if np.random.random() < 0.5:
                label = 0.0
                rel_path = variants["Cover"]
            else:
                label = 1.0
                # Choose one of the stego algorithms uniformly
                algo = np.random.choice(["JMiPOD", "JUNIWARD", "UERD"])
                rel_path = variants[algo]

        else:
            # Static Strategy (Val/Test)
            record = self.samples[idx]
            rel_path = record["file_path"]
            # Test set might not have label
            label = float(record.get("label", 0.0))

        # Load Image
        full_path = os.path.join(self.input_dir, rel_path)
        image = cv2.imread(full_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] float32
        image = image.astype(np.float32) / 255.0

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Manual conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, torch.tensor(label, dtype=torch.float32)
