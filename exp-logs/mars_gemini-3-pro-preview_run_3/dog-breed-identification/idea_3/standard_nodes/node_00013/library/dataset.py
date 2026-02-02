import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import load_metadata


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads a mapping between breed names and integer indices.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (class_to_idx (dict), classes (list))
    """
    cache_path = os.path.join(Config.working_dir, "classes.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            classes = df["breed"].tolist()
            class_to_idx = {breed: idx for idx, breed in enumerate(classes)}
            return class_to_idx, classes
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    # Load training metadata to get the full list of classes
    df_train = load_metadata("train")
    classes = sorted(df_train["breed"].unique().tolist())

    # Create DataFrame for caching
    df_cache = pd.DataFrame({"breed": classes, "idx": range(len(classes))})

    # Save to cache
    os.makedirs(Config.working_dir, exist_ok=True)
    df_cache.to_parquet(cache_path, index=False)

    class_to_idx = {breed: idx for idx, breed in enumerate(classes)}

    return class_to_idx, classes


def get_transforms(split):
    """
    Returns the torchvision transforms for a given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose(
            [
                # RandomResizedCrop is critical for generalization (Cite solution_lesson_node_00012)
                transforms.RandomResizedCrop(Config.img_size, scale=(0.08, 1.0)),
                # RandAugment for regularization
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Val/Test: Resize larger then center crop
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(Config.img_size),
                transforms.ToTensor(),
                normalize,
            ]
        )


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    """

    def __init__(self, split, class_to_idx=None, transform=None, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            class_to_idx (dict, optional): Mapping from breed name to index. Required for train/val.
            transform (callable, optional): Transform to apply to images.
            debug (bool): If True, use a small subset of data.
        """
        self.split = split
        self.transform = transform
        self.class_to_idx = class_to_idx

        # Load metadata
        self.metadata = load_metadata(split)

        # Debug mode: sample subset
        if debug:
            self.metadata = self.metadata.sample(
                n=min(len(self.metadata), Config.debug_sample_size),
                random_state=Config.seed,
            ).reset_index(drop=True)

        # Pre-compute full paths
        self.metadata["full_path"] = self.metadata["file_path"].apply(
            lambda x: os.path.join(Config.input_dir, x)
        )

        self.image_paths = self.metadata["full_path"].values
        self.ids = self.metadata["id"].values

        # Load labels if available
        if split in ["train", "val"]:
            if class_to_idx is None:
                raise ValueError("class_to_idx must be provided for train/val splits")
            self.labels = self.metadata["breed"].map(class_to_idx).values
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            # Handle missing image gracefully (though metadata validation should prevent this)
            # Return a blank image or raise error. Raising error is safer for debugging.
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR (OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        img = Image.fromarray(img)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        if self.split in ["train", "val"]:
            label = self.labels[idx]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            # For test, return image and ID
            return img, self.ids[idx]


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached class mapping.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
    """
    # Get class mapping
    class_to_idx, classes = get_class_mapping(load_cached_data=load_cached_data)

    # Verify num_classes matches Config
    if len(classes) != Config.num_classes:
        print(
            f"Warning: Discovered {len(classes)} classes, but Config.num_classes is {Config.num_classes}"
        )

    # Initialize Datasets
    train_dataset = DogDataset(
        split="train",
        class_to_idx=class_to_idx,
        transform=get_transforms("train"),
        debug=Config.debug,
    )

    val_dataset = DogDataset(
        split="val",
        class_to_idx=class_to_idx,
        transform=get_transforms("val"),
        debug=Config.debug,
    )

    test_dataset = DogDataset(
        split="test",
        class_to_idx=None,  # Not needed for test
        transform=get_transforms("test"),
        debug=Config.debug,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes
