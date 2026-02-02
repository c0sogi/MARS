import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


def normalize_image(img, mean, std):
    """
    Normalizes an image array and converts it to channel-first format.

    Args:
        img (np.ndarray): Image array of shape (H, W, 3) with values in [0, 1].
        mean (np.ndarray): Mean values for normalization.
        std (np.ndarray): Std values for normalization.

    Returns:
        np.ndarray: Normalized image of shape (3, H, W).
    """
    # Normalize: (H, W, 3) - (3,) -> (H, W, 3)
    img = (img - mean) / std
    # Transpose to Channel-First: (3, H, W)
    img = img.transpose(2, 0, 1)
    return img


class LeafDataset(Dataset):
    """
    PyTorch Dataset for loading binary leaf images.
    Implements Multi-View generation (4 rotations) for rotation invariance.
    """

    def __init__(self, metadata_path, img_size, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            img_size (int): Target spatial dimension for the images (square).
            debug (bool): If True, restricts the dataset to a small subset.
        """
        self.img_size = img_size
        self.df = pd.read_csv(metadata_path)

        if debug:
            self.df = self.df.head(20)

        self.input_dir = Config.INPUT_DIR

        # Pre-convert mean/std to numpy float32 for efficiency
        self.mean = np.array(Config.MEAN, dtype=np.float32)
        self.std = np.array(Config.STD, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            images_tensor (torch.Tensor): Shape (4, 3, H, W) containing 4 rotated views.
            label (str or int): The class label (species name) or -1 if not available.
            sample_id (int): The unique image identifier.
        """
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        # Note: Images are binary (black/white), but we load as color (3 channels)
        # to match the input requirements of pre-trained backbones (DINOv2/ConvNeXt).
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            # Safety fallback for missing files
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            # Resize to target size
            img = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to float [0, 1]
        img = img.astype(np.float32) / 255.0

        # Generate 4 rotated views for Multi-View Averaging
        # View 0: 0 degrees
        v0 = img.copy()
        # View 1: 90 degrees Counter-Clockwise
        v1 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # View 2: 180 degrees
        v2 = cv2.rotate(img, cv2.ROTATE_180)
        # View 3: 90 degrees Clockwise (270 degrees)
        v3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        views = [v0, v1, v2, v3]
        processed_views = []

        for v in views:
            # Normalize and transpose to (C, H, W)
            pv = normalize_image(v, self.mean, self.std)
            processed_views.append(pv)

        # Stack views: (4, 3, H, W)
        images_tensor = torch.tensor(np.stack(processed_views), dtype=torch.float32)

        # Extract metadata
        sample_id = row["id"]
        label = row["species"] if "species" in row else -1

        return images_tensor, label, sample_id


def load_tabular_data(metadata_path, debug=False):
    """
    Loads extracted tabular features (Margin, Shape, Texture) from the metadata CSV.

    Args:
        metadata_path (str): Path to the metadata CSV.
        debug (bool): If True, restricts data to a small subset.

    Returns:
        ids (np.ndarray): Array of sample IDs.
        X (np.ndarray): Feature matrix of shape (N, 192).
        y (np.ndarray or None): Array of target species labels (strings), or None if test set.
    """
    df = pd.read_csv(metadata_path)

    if debug:
        df = df.head(20)

    ids = df["id"].values

    # Identify feature columns based on prefixes defined in Config
    # Prefixes: 'margin', 'shape', 'texture'
    feature_cols = []
    for prefix in Config.TABULAR_COLS_PREFIXES:
        # Filter columns starting with the prefix
        # We assume the order in the CSV is correct (margin_1 ... margin_64)
        cols = [c for c in df.columns if c.startswith(prefix)]
        feature_cols.extend(cols)

    # Extract feature matrix
    X = df[feature_cols].values.astype(np.float32)

    # Extract targets if available
    y = None
    if "species" in df.columns:
        y = df["species"].values

    return ids, X, y
