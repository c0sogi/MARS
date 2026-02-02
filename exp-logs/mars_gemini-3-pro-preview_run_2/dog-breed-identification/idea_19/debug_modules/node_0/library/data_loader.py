import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config, StreamConfig


def get_class_mapping():
    """
    Generates a consistent class-to-index mapping based on the training metadata.
    Returns:
        classes (list): Sorted list of class names.
        class_to_idx (dict): Mapping from class name to integer index.
    """
    # Load train metadata to establish class mapping
    # We use the training set specifically to define the universe of classes
    df = pd.read_csv(Config.TRAIN_METADATA)
    classes = sorted(df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_transforms(stream_config: StreamConfig):
    """
    Creates the transformation pipelines for the three geometric views:
    Global, Standard, and Local.

    Args:
        stream_config (StreamConfig): Configuration object containing input size,
                                      mean, std, and interpolation mode.

    Returns:
        dict: A dictionary of torchvision.transforms.Compose objects.
    """
    # Map string interpolation to torchvision enum
    if stream_config.interpolation.lower() == "bicubic":
        interp_mode = transforms.InterpolationMode.BICUBIC
    elif stream_config.interpolation.lower() == "bilinear":
        interp_mode = transforms.InterpolationMode.BILINEAR
    else:
        interp_mode = transforms.InterpolationMode.BICUBIC

    mean = stream_config.mean
    std = stream_config.std
    size = stream_config.input_size

    # Common normalization pipeline
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    # 1. Global View: Squish to input_size
    # Captures global topology by distorting aspect ratio to fit square
    global_transform = transforms.Compose(
        [transforms.Resize((size, size), interpolation=interp_mode), norm]
    )

    # 2. Standard View: Resize and Center Crop
    # Matches standard pre-training protocols (e.g., Resize 256 -> Crop 224)
    # 224 / 0.875 = 256
    resize_size_std = int(size / 0.875)
    standard_transform = transforms.Compose(
        [
            transforms.Resize(resize_size_std, interpolation=interp_mode),
            transforms.CenterCrop(size),
            norm,
        ]
    )

    # 3. Local View: Zoom and Center Crop
    # Captures fine-grained texture details
    resize_size_local = int(size * stream_config.local_view_scale)
    local_transform = transforms.Compose(
        [
            transforms.Resize(resize_size_local, interpolation=interp_mode),
            transforms.CenterCrop(size),
            norm,
        ]
    )

    return {
        "global": global_transform,
        "standard": standard_transform,
        "local": local_transform,
    }


class DogDataset(Dataset):
    """
    Dataset class for loading dog images and generating multi-view representations.
    """

    def __init__(self, metadata_path, stream_config, mode="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train, val, or test).
            stream_config (StreamConfig): Configuration for the specific model stream.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = pd.read_csv(metadata_path)
        self.root_dir = Config.INPUT_DIR
        self.stream_config = stream_config
        self.mode = mode

        # Initialize transforms
        self.transforms = get_transforms(stream_config)

        # Handle labels
        # We always use the global training set to define class mapping
        # to ensure consistency across train and val sets.
        if mode in ["train", "val"]:
            self.classes, self.class_to_idx = get_class_mapping()
        else:
            self.classes, self.class_to_idx = None, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata contains relative path e.g., 'train/id.jpg'
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image and ensure RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately
            # For this task, we assume data integrity based on metadata checks
            image = Image.new(
                "RGB", (self.stream_config.input_size, self.stream_config.input_size)
            )

        # Apply transforms for all 3 views
        img_global = self.transforms["global"](image)
        img_standard = self.transforms["standard"](image)
        img_local = self.transforms["local"](image)

        item = {
            "id": row["id"],
            "global": img_global,
            "standard": img_standard,
            "local": img_local,
        }

        # Add label if available
        if self.mode in ["train", "val"]:
            label_str = row["breed"]
            label_idx = self.class_to_idx[label_str]
            item["label"] = torch.tensor(label_idx, dtype=torch.long)

        return item
