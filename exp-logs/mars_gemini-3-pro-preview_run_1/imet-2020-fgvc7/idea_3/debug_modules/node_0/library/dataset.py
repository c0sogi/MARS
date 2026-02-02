import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_metadata_df(mode="train", load_cached_data=True, sample_size=None):
    """
    Loads the metadata dataframe for the specified mode with caching logic.

    Args:
        mode (str): One of "train", "val", "test".
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): If provided, samples the dataframe for debugging.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    cache_filename = f"{mode}_processed.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if sample_size is not None and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(
                    drop=True
                )
            return df
        except Exception:
            # If load fails, fall through to re-process
            pass

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
    df = pd.read_csv(input_path, dtype={"id": str, "attribute_ids": str})

    # Fill NaNs in attribute_ids for train/val
    if "attribute_ids" in df.columns:
        df["attribute_ids"] = df["attribute_ids"].fillna("")

    # Ensure file_path is correct (metadata paths are relative to input dir)
    # We don't need to modify the path string here, just ensure we use it correctly in Dataset

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    # Apply sampling if requested
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)

    return df


def get_transforms(mode="train", img_size=Config.IMG_SIZE):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): "train", "val", or "test".
        img_size (int): Target image size.

    Returns:
        A.Compose: The transform composition.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                # Aggressive augmentation as per Idea
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=img_size // 10,
                    max_width=img_size // 10,
                    p=0.3,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class ArtworkDataset(Dataset):
    """
    Dataset class for loading artwork images and labels.
    """

    def __init__(self, df, mode="train", transforms=None, root_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): "train", "val", or "test".
            transforms (A.Compose): Albumentations transforms.
            root_dir (str): Root directory for images.
        """
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.root_dir = root_dir

        # Pre-process file paths to be absolute or relative to execution context
        self.file_paths = self.df["file_path"].values

        # Pre-process labels for train/val
        if self.mode != "test":
            self.labels = self.df["attribute_ids"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)

        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though verification script says 0 missing)
            # Create a black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if None provided
            t = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = t(image=image)["image"]

        # 3. Return Data based on mode
        if self.mode == "test":
            # For test, return image and id
            img_id = self.df.iloc[idx]["id"]
            return image, img_id
        else:
            # For train/val, return image and multi-hot targets
            label_str = self.labels[idx]
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

            if label_str and isinstance(label_str, str) and len(label_str.strip()) > 0:
                indices = [int(x) for x in label_str.split()]
                target[indices] = 1.0

            return image, target
