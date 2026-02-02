import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from library.config import Config


def get_transforms(stream, view):
    """
    Generates the specific transformation pipeline based on the model stream and geometric view.

    Args:
        stream (str): 'stream_a' (ConvNeXt) or 'stream_b' (ViT).
        view (str): 'global', 'standard', or 'robust'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Define Normalization (Standard ImageNet)
    # Both streams use standard ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    # Define Interpolation
    # ConvNeXt (Stream A) typically uses Bilinear in torchvision V1 recipe
    # ViT (Stream B) typically uses Bicubic in SWAG/ViT recipes
    if stream == "stream_a":
        interpolation = InterpolationMode.BILINEAR
    elif stream == "stream_b":
        interpolation = InterpolationMode.BICUBIC
    else:
        raise ValueError(f"Unknown stream: {stream}")

    # Base transform list
    t_list = []

    if view == "global":
        # Resize (Squish) to target size
        t_list.append(
            transforms.Resize(
                (Config.GLOBAL_VIEW_SIZE, Config.GLOBAL_VIEW_SIZE),
                interpolation=interpolation,
            )
        )
        t_list.append(transforms.ToTensor())
        t_list.append(normalize)

    elif view == "standard":
        # Resize to slightly larger, then Center Crop
        t_list.append(
            transforms.Resize(Config.STANDARD_RESIZE, interpolation=interpolation)
        )
        t_list.append(transforms.CenterCrop(Config.STANDARD_CROP))
        t_list.append(transforms.ToTensor())
        t_list.append(normalize)

    elif view == "robust":
        # Resize to significantly larger, then FiveCrop
        t_list.append(
            transforms.Resize(Config.ROBUST_RESIZE, interpolation=interpolation)
        )
        t_list.append(transforms.FiveCrop(Config.ROBUST_CROP))

        # FiveCrop returns a tuple of images. We need to process each crop.
        # Lambda transforms the tuple of PIL images into a tensor of shape (5, C, H, W)
        t_list.append(
            transforms.Lambda(
                lambda crops: torch.stack(
                    [normalize(transforms.ToTensor()(crop)) for crop in crops]
                )
            )
        )

    else:
        raise ValueError(f"Unknown view: {view}")

    return transforms.Compose(t_list)


class DogDataset(Dataset):
    """
    PyTorch Dataset for loading Dog images and labels.
    Handles label encoding consistency and debug sampling.
    """

    def __init__(self, metadata_path, transform=None, debug_sample_size=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train, val, or test).
            transform (callable, optional): Transform to apply to the images.
            debug_sample_size (int, optional): If set, limits dataset size for debugging.
        """
        self.metadata_path = metadata_path
        self.transform = transform
        self.debug_sample_size = debug_sample_size

        # Load the specific split metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Handle Debugging
        if self.debug_sample_size is not None and self.debug_sample_size < len(self.df):
            # Deterministic sampling for reproducibility
            self.df = self.df.sample(
                n=self.debug_sample_size, random_state=Config.SEED
            ).reset_index(drop=True)

        # Determine if this is a labeled dataset
        self.has_labels = "breed" in self.df.columns

        # Build Class Mapping (Label Encoding)
        # We must ensure the mapping is consistent across Train, Val, and Test.
        # We always build the vocabulary from the full Training set.
        self.classes = self._load_classes()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Pre-process data list for faster access
        # Format: [(full_path, label_idx, id_str), ...]
        self.samples = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            img_id = row["id"]

            label_idx = -1
            if self.has_labels:
                label_name = row["breed"]
                label_idx = self.class_to_idx.get(label_name)
                if label_idx is None:
                    raise ValueError(
                        f"Label '{label_name}' not found in training classes."
                    )

            self.samples.append((img_path, label_idx, img_id))

    def _load_classes(self):
        """
        Loads the unique classes from the training metadata to ensure consistency.
        """
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        classes = sorted(train_df["breed"].unique().tolist())
        return classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, img_id = self.samples[idx]

        try:
            # Load image and convert to RGB (handles Grayscale/RGBA)
            with Image.open(path) as img:
                img = img.convert("RGB")

                if self.transform:
                    img = self.transform(img)

                return img, label, img_id

        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a dummy tensor or raise, depending on preference.
            # For this task, raising ensures we don't silently fail on bad data.
            raise e

    def get_num_classes(self):
        return len(self.classes)
