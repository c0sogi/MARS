import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data_split="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    image_size = Config.image_size

    if data_split == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet defaults
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_and_process_df(
    csv_path, name, load_cached_data=True, debug=False, debug_size=2000
):
    """
    Loads metadata CSV, processes it, and handles caching to parquet.

    Args:
        csv_path (str): Path to the original metadata CSV.
        name (str): Name identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to run in debug mode (subset of data).
        debug_size (int): Number of samples for debug mode.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    # Construct cache filename
    debug_suffix = "_debug" if debug else ""
    cache_filename = f"{name}_processed{debug_suffix}.parquet"
    cache_path = os.path.join(Config.output_dir, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    # 2. Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    # Read CSV
    # attribute_ids might be NaN for test set or empty images, handle as string
    df = pd.read_csv(csv_path, dtype={"id": str, "attribute_ids": str})

    # Handle Missing Values
    if "attribute_ids" in df.columns:
        df["attribute_ids"] = df["attribute_ids"].fillna("")

    # Construct full file paths immediately for efficiency
    # Metadata contains relative path (e.g., "train/xxx.png")
    # Config.input_root is "./input"
    df["full_path"] = df["file_path"].apply(
        lambda x: os.path.join(Config.input_root, x)
    )

    # Debug Sampling
    if debug:
        if len(df) > debug_size:
            df = df.sample(n=debug_size, random_state=Config.seed).reset_index(
                drop=True
            )

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.
    """

    def __init__(
        self,
        csv_path,
        mode="train",
        transform=None,
        load_cached_data=True,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.mode = mode
        self.transform = transform
        self.num_classes = Config.num_classes

        # Load Data
        self.df = load_and_process_df(
            csv_path=csv_path,
            name=mode,
            load_cached_data=load_cached_data,
            debug=Config.debug,
            debug_size=Config.debug_sample_size,
        )

        # Pre-extract columns to lists/arrays for faster access in __getitem__
        self.image_paths = self.df["full_path"].values
        self.ids = self.df["id"].values

        if self.mode != "test":
            self.labels_str = self.df["attribute_ids"].values
        else:
            self.labels_str = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing/corrupt images (should not happen based on EDA)
            # Create a black image
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 3. Handle Targets
        if self.mode == "test":
            # For test, return image and ID (for submission)
            return image, self.ids[idx]
        else:
            # For train/val, return image and multi-hot encoded target
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            label_str = self.labels_str[idx]

            if label_str and len(label_str.strip()) > 0:
                # Parse space-separated integers
                indices = [int(x) for x in label_str.split()]
                target[indices] = 1.0

            return image, target
