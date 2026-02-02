import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_metadata(mode, load_cached_data=True):
    """
    Loads metadata, processing raw CSVs or loading cached Parquet files.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed metadata dataframe.
    """
    cache_filename = f"{mode}_processed.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Re-processing.")

    # 2. Process from scratch
    if mode == "train":
        input_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        input_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        input_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    # Read CSV
    # Ensure attribute_ids is read as string to preserve "0 1 2" format
    dtype_dict = {"id": str, "file_path": str}
    if mode != "test":
        dtype_dict["attribute_ids"] = str

    df = pd.read_csv(input_path, dtype=dtype_dict)

    # Resolve full image paths
    # Metadata contains relative paths like "train/xxxx.png"
    # We prepend INPUT_DIR to make them absolute for cv2
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # Handle missing labels in train/val
    if mode != "test":
        df["attribute_ids"] = df["attribute_ids"].fillna("")

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["full_path"]
        image_id = row["id"]

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing/corrupt images: return a black image
            # This prevents crashing during training
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided (Resize + ToTensor)
            base_transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = base_transform(image=image)["image"]

        # Process Labels
        if self.mode in ["train", "val"]:
            labels_str = row["attribute_ids"]
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

            if (
                labels_str
                and isinstance(labels_str, str)
                and len(labels_str.strip()) > 0
            ):
                # Parse "0 1 2" -> [0, 1, 2]
                try:
                    indices = [int(x) for x in labels_str.split()]
                    label_vec[indices] = 1.0
                except ValueError:
                    pass  # Keep as zeros if parsing fails

            return image, label_vec, image_id

        else:
            # Test mode: return dummy labels
            dummy_label = torch.zeros(self.num_classes, dtype=torch.float32)
            return image, dummy_label, image_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                # Note: Mixup/CutMix is applied in the training loop
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val/Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_loaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Dataframes
    train_df = load_metadata("train", load_cached_data)
    val_df = load_metadata("val", load_cached_data)
    test_df = load_metadata("test", load_cached_data)

    # 2. Handle Debug Mode
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Create Datasets
    train_dataset = ArtworkDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = ArtworkDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_dataset = ArtworkDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup/CutMix to have full batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
