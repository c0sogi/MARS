import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(img_size=Config.IMG_SIZE, mean=Config.MEAN, std=Config.STD):
    """
    Returns the image transformation pipeline.

    Args:
        img_size (int): Target image size (height and width).
        mean (list): Normalization mean.
        std (list): Normalization standard deviation.

    Returns:
        A.Compose: Albumentations composition.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_class_mapping(metadata_path=Config.TRAIN_METADATA, load_cached_data=True):
    """
    Generates or loads the class mapping (species -> int).

    Args:
        metadata_path (str): Path to the training metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (classes_list, class_to_idx_dict)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    classes_path = os.path.join(cache_dir, "classes.npy")
    mapping_path = os.path.join(cache_dir, "class_mapping.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(classes_path)
        and os.path.exists(mapping_path)
    ):
        classes = np.load(classes_path, allow_pickle=True)
        class_to_idx = np.load(mapping_path, allow_pickle=True).item()
        return classes.tolist(), class_to_idx

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "species" not in df.columns:
        raise ValueError("Training metadata must contain 'species' column.")

    classes = sorted(df["species"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # 3. Save to cache
    np.save(classes_path, np.array(classes))
    np.save(mapping_path, class_to_idx)

    return classes, class_to_idx


def load_tabular_data(df, split_name, class_to_idx=None, load_cached_data=True):
    """
    Extracts tabular features and targets from dataframe with caching.

    Args:
        df (pd.DataFrame): Dataframe containing features and metadata.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        class_to_idx (dict, optional): Mapping for targets. Required if 'species' is in df.
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (X, y, ids)
            X (np.ndarray): (N, 192) feature matrix.
            y (np.ndarray): (N,) target array (or None if test).
            ids (np.ndarray): (N,) id array.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    x_path = os.path.join(cache_dir, f"{split_name}_tabular_X.npy")
    y_path = os.path.join(cache_dir, f"{split_name}_tabular_y.npy")
    ids_path = os.path.join(cache_dir, f"{split_name}_tabular_ids.npy")

    # 1. Try load from cache
    if load_cached_data:
        # Check if files exist (y_path is optional for test)
        has_x = os.path.exists(x_path)
        has_ids = os.path.exists(ids_path)
        has_y = os.path.exists(y_path)

        # Determine if we expect y
        expect_y = "species" in df.columns

        if has_x and has_ids and (not expect_y or has_y):
            X = np.load(x_path)
            ids = np.load(ids_path)
            y = np.load(y_path) if has_y else None
            return X, y, ids

    # 2. Compute from scratch
    # Identify feature columns
    margin_cols = [f"margin_{i}" for i in range(1, 65)]
    shape_cols = [f"shape_{i}" for i in range(1, 65)]
    texture_cols = [f"texture_{i}" for i in range(1, 65)]
    feature_cols = margin_cols + shape_cols + texture_cols

    # Verify columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataframe: {missing_cols[:5]}...")

    X = df[feature_cols].values.astype(np.float32)
    ids = df["id"].values.astype(np.int32)

    y = None
    if "species" in df.columns:
        if class_to_idx is None:
            raise ValueError("class_to_idx must be provided for labeled data.")
        y = df["species"].map(class_to_idx).values.astype(np.int64)

    # 3. Save to cache
    np.save(x_path, X)
    np.save(ids_path, ids)
    if y is not None:
        np.save(y_path, y)

    return X, y, ids


class LeafDataset(Dataset):
    """
    Dataset class for Leaf Species Classification.
    Handles multi-view image generation (0, 90, 180, 270 degrees) and tabular data.
    """

    def __init__(
        self,
        df,
        transforms=None,
        class_to_idx=None,
        split_name="train",
        load_cached_data=True,
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe with 'file_path', 'id', and feature columns.
            transforms (A.Compose): Albumentations transforms.
            class_to_idx (dict): Mapping from species name to integer.
            split_name (str): Name of the split for caching purposes.
            load_cached_data (bool): Whether to use cached tabular features.
        """
        self.df = df
        self.transforms = transforms
        self.input_dir = Config.INPUT_DIR
        self.rotations = Config.ROTATIONS  # [0, 90, 180, 270]

        # Load tabular data and targets using the caching utility
        self.X, self.y, self.ids = load_tabular_data(
            df,
            split_name=split_name,
            class_to_idx=class_to_idx,
            load_cached_data=load_cached_data,
        )

        # Store file paths
        self.file_paths = df["file_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Read as BGR (OpenCV default)
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Generate Multi-View Stack (Canonical Rotations)
        views = []
        for angle in self.rotations:
            if angle == 0:
                rotated_img = image
            elif angle == 90:
                rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated_img = image  # Fallback

            # Apply transforms
            if self.transforms:
                augmented = self.transforms(image=rotated_img)
                views.append(augmented["image"])
            else:
                # Basic to tensor if no transforms provided
                views.append(
                    torch.from_numpy(rotated_img.transpose(2, 0, 1)).float() / 255.0
                )

        # Stack views: (4, C, H, W)
        image_stack = torch.stack(views, dim=0)

        # 3. Get Tabular Features
        tabular_features = torch.tensor(self.X[idx], dtype=torch.float32)

        # 4. Prepare Output
        sample = {
            "images": image_stack,
            "tabular": tabular_features,
            "id": self.ids[idx],
        }

        if self.y is not None:
            sample["label"] = torch.tensor(self.y[idx], dtype=torch.long)

        return sample
