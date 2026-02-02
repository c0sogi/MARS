import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train' for D4 augmentations, 'val'/'test' for simple normalization.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalize to [0, 1] as per requirements
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Normalize to [0, 1]
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def process_metadata(mode, load_cached_data=True):
    """
    Loads and processes metadata. Implements caching for the training set grouping logic.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    if mode == "train":
        cache_path = os.path.join(Config.working_dir, "train_grouped.parquet")

        # 1. Try to load cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                print(f"Loaded cached training data from {cache_path}")
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print("Processing training metadata...")
        df_raw = pd.read_csv(Config.train_csv)

        # Pivot to create one row per image_id with columns for each algo
        # Expected cols: image_id, Cover, JMiPOD, JUNIWARD, UERD
        df_pivoted = df_raw.pivot(
            index="image_id", columns="algo", values="file_path"
        ).reset_index()

        # Ensure columns exist (sanity check)
        required_cols = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
        for col in required_cols:
            if col not in df_pivoted.columns:
                raise ValueError(f"Missing algorithm column {col} in training data.")

        # Save to cache
        df_pivoted.to_parquet(cache_path, index=False)
        print(f"Saved processed training data to {cache_path}")
        return df_pivoted

    elif mode == "val":
        return pd.read_csv(Config.val_csv)

    elif mode == "test":
        return pd.read_csv(Config.test_csv)

    else:
        raise ValueError(f"Unknown mode: {mode}")


class AlaskaDataset(Dataset):
    """
    Dataset class for ALASKA2 Steganalysis.

    Modes:
    - train: Returns a pair of images (Cover, Stego) for the same content.
    - val: Returns a single image and label.
    - test: Returns a single image (no label).
    """

    def __init__(self, mode, load_cached_data=True, transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached metadata.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.mode = mode
        self.transform = transform
        self.root = Config.input_root

        # Load metadata
        self.df = process_metadata(mode, load_cached_data)

        # Debugging: Slice dataset
        if Config.debug:
            if mode == "train":
                limit = min(len(self.df), Config.debug_train_size)
                self.df = self.df.iloc[:limit].reset_index(drop=True)
            elif mode == "val":
                limit = min(len(self.df), Config.debug_val_size)
                self.df = self.df.iloc[:limit].reset_index(drop=True)
            # For test, we might keep full or slice, usually keep full for submission generation
            # but if strictly debugging pipeline, we can slice.
            # Assuming debug applies to test as well for runtime checks.
            elif mode == "test":
                limit = min(len(self.df), Config.debug_val_size)
                self.df = self.df.iloc[:limit].reset_index(drop=True)

        print(f"[{mode.upper()}] Dataset initialized. Size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.mode == "train":
            # Dynamic Pairing Strategy
            # 1. Load Cover
            cover_path = os.path.join(self.root, row["Cover"])
            cover_img = self._load_image(cover_path)

            # 2. Select Random Stego Variant
            stego_algos = ["JMiPOD", "JUNIWARD", "UERD"]
            selected_algo = np.random.choice(stego_algos)
            stego_path = os.path.join(self.root, row[selected_algo])
            stego_img = self._load_image(stego_path)

            # 3. Apply Transforms
            # Note: We apply transforms independently to learn robust features
            if self.transform:
                res_cover = self.transform(image=cover_img)
                cover_tensor = res_cover["image"]

                res_stego = self.transform(image=stego_img)
                stego_tensor = res_stego["image"]
            else:
                cover_tensor = (
                    torch.from_numpy(cover_img.transpose(2, 0, 1)).float() / 255.0
                )
                stego_tensor = (
                    torch.from_numpy(stego_img.transpose(2, 0, 1)).float() / 255.0
                )

            # 4. Stack and Return
            # Returns: (2, C, H, W), (2,)
            images = torch.stack([cover_tensor, stego_tensor])
            labels = torch.tensor([0.0, 1.0], dtype=torch.float32)

            return images, labels

        else:
            # Val / Test Mode
            file_path = os.path.join(self.root, row["file_path"])
            image = self._load_image(file_path)

            if self.transform:
                res = self.transform(image=image)
                image_tensor = res["image"]
            else:
                image_tensor = (
                    torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                )

            if self.mode == "test":
                # Test: Return image and ID (for submission file creation)
                return image_tensor, row["image_id"]
            else:
                # Val: Return image and label
                label = torch.tensor(row["label"], dtype=torch.float32)
                return image_tensor, label

    def _load_image(self, path):
        """Helper to load and convert image."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
