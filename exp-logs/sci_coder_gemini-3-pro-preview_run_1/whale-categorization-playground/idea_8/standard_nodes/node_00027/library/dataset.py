import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library import config
from library import transforms


def get_class_mapping(load_cached_data=True):
    """
    Retrieves the list of unique whale IDs (classes) to create a consistent
    Label -> Index mapping. Implements caching to ensure reproducibility
    and consistency between training and inference.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.
                                 If False or load fails, recomputes.

    Returns:
        np.ndarray: Sorted array of unique class names (strings).
    """
    cache_path = os.path.join(config.WORKING_DIR, "classes.npy")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path, allow_pickle=True)
            return classes
        except Exception:
            pass  # Fallback to recomputing

    # 2. Compute from scratch
    # We always derive the master class list from the full training set
    df = pd.read_csv(config.TRAIN_CSV)

    # Get unique IDs and sort them for determinism
    unique_ids = df["Id"].unique()
    classes = np.sort(unique_ids)

    # 3. Save to cache
    np.save(cache_path, classes)

    return classes


class WhaleDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        transform=None,
        class_mapping=None,
        load_cached_data=True,
    ):
        """
        Args:
            csv_file (str): Path to the metadata CSV file (train, val, or test).
            img_dir (str): Path to the directory containing images.
            transform (albumentations.Compose): Augmentation pipeline.
            class_mapping (np.ndarray, optional): Array of class names. If None,
                it is loaded/generated using get_class_mapping.
            load_cached_data (bool): Whether to use cached class mapping.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Determine if this is a labeled dataset (Train/Val) or Unlabeled (Test)
        self.has_labels = "Id" in self.df.columns

        # Setup Label Encoding
        if class_mapping is None:
            self.classes = get_class_mapping(load_cached_data=load_cached_data)
        else:
            self.classes = class_mapping

        # Create a dictionary for fast lookup: Label Str -> Index Int
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image"]

        # Construct full image path
        # The metadata file_path column is relative to input dir, but we can also
        # just join img_dir with the filename if img_dir is specific (e.g. input/train)
        # Given the config, img_dir is like "./input/train".
        img_path = os.path.join(self.img_dir, img_name)

        # Load Image
        # cv2.imread returns BGR. Returns None if file missing/corrupt.
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing images to avoid crashing, though metadata should be clean.
            # Create a black image of default size
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided: ToTensor
            # (Assuming transform is always provided in this pipeline, but for safety)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.has_labels:
            label_str = row["Id"]
            # Map string label to integer index
            # If a label is not in the mapping (should not happen with correct split),
            # we might handle it, but here we assume consistency.
            label_idx = self.label_to_idx[label_str]
            return image, label_idx
        else:
            # For test set, return image and the filename (ID) to map predictions later
            return image, img_name
