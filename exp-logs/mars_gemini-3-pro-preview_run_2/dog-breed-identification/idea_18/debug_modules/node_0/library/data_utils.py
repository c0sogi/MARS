import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import pandas as pd
import numpy as np
import library.config as config

# Standard ImageNet Normalization Constants
# These are standard for both IMAGENET1K_V1 and SWAG weights in torchvision
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_interpolation_mode(mode_str):
    """
    Maps string interpolation mode to torchvision InterpolationMode enum.
    """
    if mode_str.lower() == "bicubic":
        return InterpolationMode.BICUBIC
    elif mode_str.lower() == "bilinear":
        return InterpolationMode.BILINEAR
    elif mode_str.lower() == "nearest":
        return InterpolationMode.NEAREST
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode_str}")


def get_label_map(train_csv_path=config.TRAIN_CSV):
    """
    Generates a mapping from breed name to integer index.
    Crucially, this sorts breeds alphabetically to match the submission format.

    Returns:
        label_to_idx (dict): {'breed_name': index, ...}
        class_names (list): List of breed names in order.
    """
    df = pd.read_csv(train_csv_path)
    # Sort unique breeds alphabetically
    unique_breeds = sorted(df["breed"].unique().tolist())

    label_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
    return label_to_idx, unique_breeds


def build_stream_transforms(stream_config):
    """
    Constructs the dictionary of transform pipelines for a specific stream.

    Args:
        stream_config (dict): Configuration dict (e.g., config.STREAM_A)

    Returns:
        dict: {'global': transform, 'standard': transform, 'local': transform}
    """
    interp_mode = get_interpolation_mode(stream_config["interpolation"])
    views_config = stream_config["views"]

    # Common normalization (ToTensor + Normalize)
    # Applied at the end of each pipeline
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    transform_dict = {}

    # 1. Global View: Resize to Input Size (Squish)
    # config: "resize_dims": (H, W)
    global_cfg = views_config["global"]
    transform_dict["global"] = transforms.Compose(
        [
            transforms.Resize(global_cfg["resize_dims"], interpolation=interp_mode),
            *common_transforms,
        ]
    )

    # 2. Standard View: Resize and Center Crop
    # config: "resize_size": int, "crop_size": int
    std_cfg = views_config["standard"]
    transform_dict["standard"] = transforms.Compose(
        [
            transforms.Resize(std_cfg["resize_size"], interpolation=interp_mode),
            transforms.CenterCrop(std_cfg["crop_size"]),
            *common_transforms,
        ]
    )

    # 3. Local View: Resize (Zoom) and Center Crop
    # config: "resize_size": int (larger), "crop_size": int
    local_cfg = views_config["local"]
    transform_dict["local"] = transforms.Compose(
        [
            transforms.Resize(local_cfg["resize_size"], interpolation=interp_mode),
            transforms.CenterCrop(local_cfg["crop_size"]),
            *common_transforms,
        ]
    )

    return transform_dict


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Loads images and applies Multi-View transformations.
    """

    def __init__(
        self, metadata_path, transform_dict, label_to_idx=None, return_label=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
            transform_dict (dict): Dictionary of transforms {'global': ..., 'standard': ..., 'local': ...}.
            label_to_idx (dict, optional): Mapping from breed name to index. Required if return_label=True.
            return_label (bool): Whether to return the label index.
        """
        self.df = pd.read_csv(metadata_path)
        self.transform_dict = transform_dict
        self.label_to_idx = label_to_idx
        self.return_label = return_label
        self.input_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        rel_path = row["file_path"]

        # Construct full path
        img_path = os.path.join(self.input_dir, rel_path)

        # Load Image
        try:
            # Convert to RGB to handle grayscale or RGBA images consistently
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately.
            # For this task, we assume data integrity based on metadata checks.
            # Create a black image of standard size to prevent crash
            image = Image.new("RGB", (224, 224))

        # Apply Transforms for each view
        views = {}
        for view_name, transform_pipeline in self.transform_dict.items():
            views[view_name] = transform_pipeline(image)

        # Handle Label
        label_idx = -1
        if self.return_label:
            if "breed" in row and self.label_to_idx:
                breed_name = row["breed"]
                label_idx = self.label_to_idx[breed_name]
            else:
                # Fallback if label is requested but not found (should not happen for train/val)
                label_idx = -1

        return {
            "id": img_id,
            "views": views,  # Dict: {'global': tensor, 'standard': tensor, 'local': tensor}
            "label": label_idx,
        }
