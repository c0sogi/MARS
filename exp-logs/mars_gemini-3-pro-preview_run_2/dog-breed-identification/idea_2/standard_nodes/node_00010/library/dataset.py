import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from library import config, utils


def get_class_mappings(csv_path):
    """
    Generates class-to-index and index-to-class mappings from a CSV file.
    Ensures deterministic mapping by sorting unique breed names.

    Args:
        csv_path (str): Path to the CSV file containing a 'breed' column.

    Returns:
        tuple: (class_to_idx, idx_to_class) dictionaries.
    """
    df = pd.read_csv(csv_path)
    if "breed" not in df.columns:
        raise ValueError(f"Column 'breed' not found in {csv_path}")

    # Sort unique breeds to ensure deterministic mapping
    unique_breeds = sorted(df["breed"].unique())
    class_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
    idx_to_class = {idx: breed for idx, breed in enumerate(unique_breeds)}

    return class_to_idx, idx_to_class


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Handles dual-stream preprocessing for CNN and ViT backbones.
    """

    def __init__(self, metadata_path, transforms=None, class_to_idx=None, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train, val, or test).
            transforms (dict, optional): Dictionary containing transforms for 'cnn' and 'vit' keys.
            class_to_idx (dict, optional): Mapping from breed name to integer label.
                                           Should be provided for validation/test sets to ensure consistency.
            debug (bool or int, optional): If True or int > 0, limits the dataset size for debugging.
        """
        self.metadata_path = metadata_path
        self.transforms = transforms
        self.class_to_idx = class_to_idx

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Debugging: Limit dataset size
        if debug:
            limit = 100 if isinstance(debug, bool) else int(debug)
            self.df = self.df.iloc[:limit]

        # Determine if this is a labeled dataset
        self.has_labels = "breed" in self.df.columns

        # If labels exist but no mapping provided, generate it (mostly for standalone training usage)
        if self.has_labels and self.class_to_idx is None:
            unique_breeds = sorted(self.df["breed"].unique())
            self.class_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]

        # Construct full image path
        # Metadata 'file_path' is relative to input directory (e.g., 'train/xxx.jpg')
        rel_path = row["file_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for compatibility with torchvision/timm transforms
        img_pil = Image.fromarray(img)

        # Initialize outputs
        cnn_img = img_pil
        vit_img = img_pil

        # Apply transforms if provided
        if self.transforms:
            if "cnn" in self.transforms:
                cnn_img = self.transforms["cnn"](img_pil)
            if "vit" in self.transforms:
                vit_img = self.transforms["vit"](img_pil)

        # Handle Label
        label = -1
        if self.has_labels:
            breed = row["breed"]
            if self.class_to_idx and breed in self.class_to_idx:
                label = self.class_to_idx[breed]
            else:
                # If label is missing from mapping (should not happen in valid pipeline), return -1
                label = -1

        return {
            "cnn_img": cnn_img,
            "vit_img": vit_img,
            "label": torch.tensor(label, dtype=torch.long),
            "id": image_id,
        }
