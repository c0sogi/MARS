import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import INPUT_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, MODELS


def get_transforms(weights):
    """
    Returns the preprocessing transforms associated with the specific model weights.

    Args:
        weights: A torchvision weights object (e.g., ConvNeXt_Large_Weights.IMAGENET1K_V1).

    Returns:
        A callable transform.
    """
    return weights.transforms()


class DogDataset(Dataset):
    """
    A PyTorch Dataset for loading dog images with model-specific transforms.
    """

    def __init__(self, csv_path, transform=None, class_to_idx=None, is_test=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (callable, optional): Transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required for train/val.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        rel_path = row["file_path"]

        # Construct full image path
        img_path = os.path.join(INPUT_DIR, rel_path)

        # Load image and convert to RGB (ensure 3 channels)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # In a real scenario, we might log this. For now, raise to fail fast.
            raise RuntimeError(f"Failed to load image at {img_path}") from e

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # For test set, return image, dummy label, and ID
            return image, -1, img_id
        else:
            # For train/val, return image, label index, and ID
            breed = row["breed"]
            label = self.class_to_idx[breed]
            return image, label, img_id


def create_dataloaders(model_name, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        model_name (str): The key corresponding to the model in library.config.MODELS.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
            - classes is a sorted list of breed names.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not defined in library.config.MODELS")

    # Retrieve model-specific configuration
    model_conf = MODELS[model_name]
    weights = model_conf["weights"]

    # Get the exact transforms required by the pre-trained weights
    transform = get_transforms(weights)

    # Establish the class mapping using the training data
    # This ensures consistency across all datasets
    train_df = pd.read_csv(TRAIN_CSV)
    classes = sorted(train_df["breed"].unique().tolist())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # Instantiate Datasets
    train_dataset = DogDataset(
        csv_path=TRAIN_CSV,
        transform=transform,
        class_to_idx=class_to_idx,
        is_test=False,
    )

    val_dataset = DogDataset(
        csv_path=VAL_CSV, transform=transform, class_to_idx=class_to_idx, is_test=False
    )

    test_dataset = DogDataset(
        csv_path=TEST_CSV, transform=transform, class_to_idx=None, is_test=True
    )

    # Instantiate DataLoaders
    # Note: Shuffle is True for training, False for val/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes
