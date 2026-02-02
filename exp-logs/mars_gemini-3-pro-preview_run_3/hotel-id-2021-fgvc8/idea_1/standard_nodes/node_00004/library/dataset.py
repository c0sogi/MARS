import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config
from library.utils import get_label_encoder


def get_transforms(split="train"):
    """
    Returns the image transformations for the given split.

    Strategy:
    - Train: RandomResizedCrop + HorizontalFlip for augmentation (Cite solution_lesson_node_00001)
    - Val/Test: Resize + CenterCrop (Deterministic)
    """
    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    else:
        transform_list = [
            transforms.Resize((Config.RESIZE_SIZE, Config.RESIZE_SIZE)),
            transforms.CenterCrop(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    return transforms.Compose(transform_list)


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Handles loading images, processing metadata, and encoding labels.
    """

    def __init__(
        self,
        csv_path,
        root_dir,
        label_encoder=None,
        transform=None,
        is_test=False,
        load_cached_data=True,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the images (e.g., ./input).
            label_encoder (LabelEncoder, optional): Encoder to convert hotel_ids to indices.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test set (returns image ID instead of label).
            load_cached_data (bool): Whether to load processed dataframe from cache.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_encoder = label_encoder

        # Determine cache path based on csv filename
        csv_name = os.path.basename(csv_path).replace(".csv", "")
        cache_filename = f"dataset_{csv_name}_processed.parquet"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        self.df = None

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                self.df = pd.read_parquet(cache_path)
                # Verify required columns exist
                required_cols = ["file_path", "image"]
                if not self.is_test:
                    required_cols.append("label_idx")

                if all(col in self.df.columns for col in required_cols):
                    # Cache is valid
                    pass
                else:
                    # Cache is invalid structure, reload
                    self.df = None
            except Exception:
                self.df = None

        # 2. Compute from scratch if not loaded
        if self.df is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Metadata file not found at {csv_path}")

            df_raw = pd.read_csv(csv_path)

            # Process labels if not test
            if not self.is_test:
                if "hotel_id" not in df_raw.columns:
                    raise ValueError("Column 'hotel_id' missing in metadata.")

                if self.label_encoder is None:
                    # For training/validation, a label encoder is required
                    raise ValueError(
                        "LabelEncoder must be provided for training/validation sets."
                    )

                # Transform labels to integers
                # The encoder handles the mapping.
                df_raw["label_idx"] = self.label_encoder.transform(
                    df_raw["hotel_id"].values
                )

                # Filter out unknown labels if any (-1 indicates unknown)
                valid_mask = df_raw["label_idx"] != -1
                df_raw = df_raw[valid_mask].reset_index(drop=True)

            self.df = df_raw

            # 3. Save to cache
            try:
                self.df.to_parquet(cache_path, index=False)
            except Exception:
                # Non-critical failure if cache cannot be written
                pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct image path
        # file_path in metadata is relative (e.g. train_images/0/xxx.jpg)
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        # Using cv2 for robustness (faster than PIL for large reads), then converting
        image = cv2.imread(img_path)

        if image is None:
            # Handle missing/corrupt image by creating a black placeholder
            # This ensures the dataloader doesn't crash
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and image filename (for submission mapping)
            return image, row["image"]
        else:
            # Return image and label index
            label = row["label_idx"]
            return image, torch.tensor(label, dtype=torch.long)
