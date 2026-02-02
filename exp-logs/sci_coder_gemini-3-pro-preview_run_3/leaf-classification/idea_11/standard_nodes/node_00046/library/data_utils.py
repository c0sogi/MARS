import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from library.config import Config

# ==========================================
# Constants
# ==========================================
# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# Utility Functions
# ==========================================


def load_and_preprocess_image(image_path, target_size):
    """
    Loads an image from disk, converts it to RGB, resizes it to the target size,
    and applies ImageNet normalization.

    Args:
        image_path (str): Path to the image file.
        target_size (int): The height and width to resize the image to.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (3, target_size, target_size).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image with OpenCV: {image_path}")

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for compatibility with torchvision transforms
    img_pil = Image.fromarray(img)

    # Define Transform Pipeline
    # Bicubic interpolation is preferred for preserving details in leaf margins
    transform = transforms.Compose(
        [
            transforms.Resize(
                (target_size, target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transform(img_pil)


def get_four_views(image_tensor):
    """
    Generates 4 rotated views (0, 90, 180, 270 degrees) of the input image tensor.
    This creates the 'expanded manifold' views for training and centroid calculation.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (C, H, W).

    Returns:
        torch.Tensor: Stacked tensor of shape (4, C, H, W).
    """
    views = []
    # 0 degrees (Original)
    views.append(image_tensor)
    # 90 degrees
    views.append(F.rotate(image_tensor, 90))
    # 180 degrees
    views.append(F.rotate(image_tensor, 180))
    # 270 degrees
    views.append(F.rotate(image_tensor, 270))

    # Stack along a new dimension 0
    return torch.stack(views)


# ==========================================
# Dataset Class
# ==========================================


class LeafImageDataset(Dataset):
    """
    PyTorch Dataset for Leaf Classification.

    Provides:
        - Multi-view image tensors for DINOv2 (Global Geometry)
        - Multi-view image tensors for ConvNeXt (Local Texture)
        - Raw tabular features (Margin, Shape, Texture)

    Returns dictionary with keys:
        - 'id': Image ID
        - 'dino_views': (4, 3, 518, 518) tensor
        - 'convnext_views': (4, 3, 384, 384) tensor
        - 'tabular': (192,) tensor
        - 'label': Species label (str) if available
    """

    def __init__(self, metadata_path, return_labels=True):
        super().__init__()
        self.metadata_path = metadata_path
        self.return_labels = return_labels

        # Load metadata CSV
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Debug Mode: Subset data to speed up development loop
        if Config.DEBUG:
            print(f"[DEBUG] Subsetting dataset to {Config.DEBUG_SAMPLES} samples.")
            self.df = self.df.iloc[: Config.DEBUG_SAMPLES].reset_index(drop=True)

        # Define Tabular Feature Columns
        # The dataset provides 64 features each for margin, shape, and texture
        self.margin_cols = [f"margin{i}" for i in range(1, 65)]
        self.shape_cols = [f"shape{i}" for i in range(1, 65)]
        self.texture_cols = [f"texture{i}" for i in range(1, 65)]
        self.feature_cols = self.margin_cols + self.shape_cols + self.texture_cols

        # Validate that tabular columns exist in the dataframe
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Metadata is missing feature columns: {missing[:3]}...")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # row['file_path'] is relative (e.g., 'images/12.jpg'), Config.INPUT_DIR is './input'
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # 1. Global Geometry Stream (DINOv2)
        # Load and resize to 518x518 as required by ViT-Large
        dino_tensor = load_and_preprocess_image(image_path, Config.DINO_IMG_SIZE)
        dino_views = get_four_views(dino_tensor)  # Shape: (4, 3, 518, 518)

        # 2. Local Texture Stream (ConvNeXt)
        # Load and resize to 384x384
        convnext_tensor = load_and_preprocess_image(
            image_path, Config.CONVNEXT_IMG_SIZE
        )
        convnext_views = get_four_views(convnext_tensor)  # Shape: (4, 3, 384, 384)

        # 3. Tabular Features
        # Extract and convert to float32 tensor
        tabular_data = row[self.feature_cols].values.astype(np.float32)
        tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32)

        # 4. Construct Result Dictionary
        sample = {
            "id": row["id"],
            "dino_views": dino_views,
            "convnext_views": convnext_views,
            "tabular": tabular_tensor,
        }

        # 5. Add Label if requested and available (for training/validation)
        if self.return_labels and "species" in row:
            sample["label"] = row["species"]

        return sample
