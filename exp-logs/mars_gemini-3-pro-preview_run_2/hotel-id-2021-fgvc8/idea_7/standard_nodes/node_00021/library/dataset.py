import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train' or 'valid'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet stats
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads a mapping from hotel_id (int) to class index (0..N-1).
    Caches the result as a parquet file to ensure determinism and speed.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        dict: Mapping {hotel_id: class_index}
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_mapping.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_map = pd.read_parquet(cache_path)
            mapping = dict(zip(df_map["hotel_id"], df_map["class_idx"]))
            return mapping
        except Exception as e:
            print(f"Failed to load class mapping cache: {e}. Recomputing...")

    # 2. Compute from scratch
    # We use the training metadata to define the universe of classes.
    # Validation classes are a subset of training classes.
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    unique_ids = sorted(df_train["hotel_id"].unique())

    mapping = {hid: idx for idx, hid in enumerate(unique_ids)}

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_map = pd.DataFrame(
        {"hotel_id": list(mapping.keys()), "class_idx": list(mapping.values())}
    )
    df_map.to_parquet(cache_path, index=False)

    return mapping


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Handles loading images, converting colorspace, and applying transforms.
    """

    def __init__(
        self,
        csv_path,
        transform=None,
        class_mapping=None,
        is_test=False,
        data_root=Config.INPUT_DIR,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (A.Compose): Albumentations transforms.
            class_mapping (dict): Mapping from hotel_id to class index. Required for train/val.
            is_test (bool): If True, ignores target labels.
            data_root (str): Root directory for images.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_mapping = class_mapping
        self.is_test = is_test
        self.data_root = data_root

        # Pre-process labels if not testing
        if not self.is_test:
            if self.class_mapping is None:
                raise ValueError(
                    "class_mapping must be provided for training/validation"
                )

            # Map hotel_ids to contiguous indices
            # We assume all hotel_ids in this df exist in class_mapping
            # (guaranteed by metadata generation logic)
            self.labels = [self.class_mapping[hid] for hid in self.df["hotel_id"]]
        else:
            self.labels = [-1] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata file_path is relative to input dir (e.g., "train_images/0/img.jpg")
        file_path = row["file_path"]
        full_path = os.path.join(self.data_root, file_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata validation should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        target = self.labels[idx]
        image_name = row["image"]

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.long),
            "image_name": image_name,
        }
