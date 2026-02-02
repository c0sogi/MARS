import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
)


# Set seeds for reproducibility
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed) # random is not imported, but numpy covers most needs here


set_seed()


def get_class_mapping(metadata_path=TRAIN_METADATA_PATH):
    """
    Generates a consistent mapping from breed name to integer index.
    Reads the training metadata to ensure all classes are covered.
    """
    df = pd.read_csv(metadata_path)
    unique_breeds = sorted(df["breed"].unique().tolist())
    breed_to_idx = {breed: i for i, breed in enumerate(unique_breeds)}
    idx_to_breed = {i: breed for i, breed in enumerate(unique_breeds)}
    return breed_to_idx, idx_to_breed


def get_transforms(stream_config):
    """
    Creates a dictionary of transforms for 'global', 'standard', and 'local' views
    based on the stream configuration.

    Args:
        stream_config (dict): Configuration dictionary for the specific stream
                              (e.g., STREAMS['stream_a']).

    Returns:
        dict: A dictionary where keys are view names and values are torchvision transforms.
    """
    library = stream_config.get("library", "torchvision")

    # Select interpolation mode
    if library == "timm":
        interpolation = InterpolationMode.BICUBIC
    else:
        interpolation = InterpolationMode.BILINEAR

    view_configs = stream_config.get("views", {})
    transform_dict = {}

    # Base normalization transform
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    for view_name, settings in view_configs.items():
        ops = []

        # 1. Resize
        # If resize is a tuple, it's (H, W) -> Squish
        # If resize is an int, it's the smaller edge -> Keep Aspect Ratio
        resize_param = settings.get("resize")
        ops.append(transforms.Resize(resize_param, interpolation=interpolation))

        # 2. Crop (if specified)
        crop_size = settings.get("crop")
        if crop_size is not None:
            ops.append(transforms.CenterCrop(crop_size))

        # 3. ToTensor and Normalize
        ops.append(transforms.ToTensor())
        ops.append(normalize)

        transform_dict[view_name] = transforms.Compose(ops)

    return transform_dict


class MultiViewDataset(Dataset):
    """
    Dataset that returns multiple geometric views of the same image.
    """

    def __init__(self, metadata_path, transform_dict, breed_to_idx=None, is_test=False):
        """
        Args:
            metadata_path (str): Path to the CSV file containing metadata.
            transform_dict (dict): Dictionary of transforms for each view.
            breed_to_idx (dict, optional): Mapping from breed string to integer. Required if not is_test.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = pd.read_csv(metadata_path)
        self.transform_dict = transform_dict
        self.is_test = is_test
        self.breed_to_idx = breed_to_idx

        # Pre-check file existence to avoid runtime errors
        # (Optional optimization, but good for debugging)
        self.df["full_path"] = self.df["file_path"].apply(
            lambda x: os.path.join(INPUT_DIR, x)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_path"]
        img_id = row["id"]

        # Load Image
        try:
            # Open and convert to RGB (handles Grayscale or RGBA)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (though analysis showed none)
            print(f"Error loading image {img_path}: {e}")
            # Return a black image of standard size to prevent crash
            image = Image.new("RGB", (224, 224))

        # Apply Transforms for each view
        views = {}
        for view_name, transform in self.transform_dict.items():
            views[view_name] = transform(image)

        # Handle Label
        if self.is_test:
            target = -1  # Dummy target for test set
        else:
            breed = row["breed"]
            target = self.breed_to_idx[breed]

        return views, target, img_id


def get_dataloaders(stream_config, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Creates DataLoaders for Train, Validation, and Test sets for a specific stream.

    Args:
        stream_config (dict): Configuration for the stream.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader, breed_to_idx)
    """
    # 1. Prepare Transforms
    transforms_dict = get_transforms(stream_config)

    # 2. Prepare Class Mapping
    breed_to_idx, _ = get_class_mapping(TRAIN_METADATA_PATH)

    # 3. Create Datasets
    train_dataset = MultiViewDataset(
        metadata_path=TRAIN_METADATA_PATH,
        transform_dict=transforms_dict,
        breed_to_idx=breed_to_idx,
        is_test=False,
    )

    val_dataset = MultiViewDataset(
        metadata_path=VAL_METADATA_PATH,
        transform_dict=transforms_dict,
        breed_to_idx=breed_to_idx,
        is_test=False,
    )

    test_dataset = MultiViewDataset(
        metadata_path=TEST_METADATA_PATH,
        transform_dict=transforms_dict,
        breed_to_idx=None,
        is_test=True,
    )

    # 4. Create DataLoaders
    # Shuffle Train, but not Val/Test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, breed_to_idx
