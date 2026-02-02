import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the specified mode.
    Strictly follows the geometric pipeline: Resize(256) -> CenterCrop(224).
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        transform = T.Compose(
            [
                # Resize the smaller edge to RESIZE_SIZE, maintaining aspect ratio
                T.Resize(Config.RESIZE_SIZE),
                # Center crop to the input size expected by the model
                T.CenterCrop(Config.CROP_SIZE),
                # Augmentation
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test
        transform = T.Compose(
            [
                T.Resize(Config.RESIZE_SIZE),
                T.CenterCrop(Config.CROP_SIZE),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    return transform


def get_label_mapping(load_cached_data=True):
    """
    Generates or loads the class-to-index mapping.
    Ensures alphabetical sorting for consistency with submission format.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "label_map.npy")

    if load_cached_data and os.path.exists(cache_path):
        classes = np.load(cache_path, allow_pickle=True)
    else:
        # Load training metadata to get all unique breeds
        if not os.path.exists(Config.TRAIN_CSV):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_CSV}"
            )

        df = pd.read_csv(Config.TRAIN_CSV)
        classes = sorted(df["breed"].unique())
        classes = np.array(classes)

        # Save to cache
        np.save(cache_path, classes)

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return class_to_idx, classes


def load_data(mode="train", load_cached_data=True):
    """
    Loads the metadata dataframe for the given mode.
    Implements caching using parquet files.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_processed.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception:
            # If load fails, proceed to re-process
            pass

    # 2. Process data from scratch
    if mode == "train":
        source_path = Config.TRAIN_CSV
    elif mode == "val":
        source_path = Config.VAL_CSV
    elif mode == "test":
        source_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)

    # For train and val, encode labels
    if mode in ["train", "val"]:
        class_to_idx, _ = get_label_mapping(load_cached_data=load_cached_data)

        # Map breeds to integers
        # Using map is faster and cleaner
        if "breed" not in df.columns:
            raise KeyError(f"Column 'breed' missing in {mode} metadata")

        df["label"] = df["breed"].map(class_to_idx)

        # Check for any unmapped labels
        if df["label"].isnull().any():
            raise ValueError(
                f"Found unknown breeds in {mode} set that are not in the training set."
            )

        df["label"] = df["label"].astype(int)

    # 3. Save to cache
    df.to_parquet(cache_file, index=False)

    return df


class DogDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, [label]).
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Pre-check columns
        if "file_path" not in df.columns:
            raise ValueError("DataFrame must contain 'file_path' column")
        if mode in ["train", "val"] and "label" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'label' column for train/val modes"
            )
        if mode == "test" and "id" not in df.columns:
            raise ValueError("DataFrame must contain 'id' column for test mode")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input directory (e.g., "train/xxx.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using PIL to ensure compatibility with torchvision transforms
        # and correct handling of RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback or error handling; usually datasets are clean but good to be safe
            # Create a black image if loading fails to prevent crashing
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (Config.RESIZE_SIZE, Config.RESIZE_SIZE))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test, return image and ID for submission mapping
            img_id = row["id"]
            return image, img_id
