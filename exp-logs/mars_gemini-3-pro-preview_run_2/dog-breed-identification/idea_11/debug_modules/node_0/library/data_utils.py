import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import numpy as np
from library import config

# ==========================================
# Constants
# ==========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================
# Helper Functions
# ==========================================
def get_class_mapping():
    """
    Generates class-to-index and index-to-class mappings based on the training data.
    Sorts breeds alphabetically to ensure deterministic ordering.
    """
    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    classes = sorted(df["breed"].unique().tolist())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for i, cls in enumerate(classes)}
    return class_to_idx, idx_to_class


def get_transforms(view_type):
    """
    Returns a callable transform function for the specified view type.
    The returned function takes a PIL Image and returns a Tensor of shape (V, C, H, W).
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if view_type == "global":
        # View 1: Global View (Shape)
        # Resize to 224x224 (Squish), then Flip TTA
        # Output: (2, 3, 224, 224)
        def transform_global(img):
            # Resize ignoring aspect ratio
            img = F.resize(img, config.GLOBAL_VIEW_RESIZE)

            # Generate Horizontal Flip
            img_flip = F.hflip(img)

            # Convert to Tensor and Normalize
            t_orig = normalize(F.to_tensor(img))
            t_flip = normalize(F.to_tensor(img_flip))

            return torch.stack([t_orig, t_flip])

        return transform_global

    elif view_type == "standard":
        # View 2: Standard View (Context)
        # Resize small edge to 232, Center Crop 224, then Flip TTA
        # Output: (2, 3, 224, 224)
        def transform_standard(img):
            # Resize maintaining aspect ratio
            img = F.resize(img, config.STANDARD_VIEW_RESIZE)
            # Center Crop
            img = F.center_crop(img, config.STANDARD_VIEW_CROP)

            # Generate Horizontal Flip
            img_flip = F.hflip(img)

            # Convert to Tensor and Normalize
            t_orig = normalize(F.to_tensor(img))
            t_flip = normalize(F.to_tensor(img_flip))

            return torch.stack([t_orig, t_flip])

        return transform_standard

    elif view_type == "local":
        # View 3: Robust Local View (Texture & Spatial Aggregation)
        # Resize small edge to 288, TenCrop 224 (5 crops + 5 flips)
        # Output: (10, 3, 224, 224)
        def transform_local(img):
            # Resize maintaining aspect ratio
            img = F.resize(img, config.LOCAL_VIEW_RESIZE)

            # TenCrop returns tuple of 10 PIL images
            # (TL, TR, BL, BR, Center, TL_flip, TR_flip, BL_flip, BR_flip, Center_flip)
            crops = transforms.TenCrop(config.LOCAL_VIEW_CROP)(img)

            # Convert each to Tensor and Normalize
            tensors = [normalize(F.to_tensor(crop)) for crop in crops]

            return torch.stack(tensors)

        return transform_local

    else:
        raise ValueError(f"Unknown view_type: {view_type}")


# ==========================================
# Dataset Class
# ==========================================
class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Loads images and applies the specific multi-view transform.
    """

    def __init__(self, metadata_path, view_type, class_to_idx=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            view_type (str): 'global', 'standard', or 'local'.
            class_to_idx (dict, optional): Mapping from breed name to integer index.
        """
        self.df = pd.read_csv(metadata_path)
        self.transform = get_transforms(view_type)
        self.input_dir = config.INPUT_DIR
        self.class_to_idx = class_to_idx

        # Check if labels are present in the CSV
        self.has_labels = "breed" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains relative path (e.g., 'train/id.jpg')
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # In case of read error, we raise it.
            # (Analysis showed 0 missing files, so this should be rare)
            raise IOError(f"Failed to load image at {img_path}: {e}")

        # Apply Transform
        # Returns tensor of shape (V, 3, H, W)
        image_tensor = self.transform(image)

        # Process Label
        label = -1
        if self.has_labels:
            breed = row["breed"]
            if self.class_to_idx:
                label = self.class_to_idx.get(breed, -1)
            else:
                # If no mapping provided, we cannot return a valid integer label
                # This might happen if using dataset just for visualization or raw access
                pass

        return image_tensor, label, row["id"]


def get_dataset(split, view_type):
    """
    Factory function to create a DogDataset for a specific split and view.

    Args:
        split (str): 'train', 'val', or 'test'.
        view_type (str): 'global', 'standard', or 'local'.

    Returns:
        DogDataset instance.
    """
    # Load class mapping
    class_to_idx, _ = get_class_mapping()

    if split == "train":
        path = config.TRAIN_METADATA_PATH
    elif split == "val":
        path = config.VAL_METADATA_PATH
    elif split == "test":
        path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    return DogDataset(path, view_type, class_to_idx)
