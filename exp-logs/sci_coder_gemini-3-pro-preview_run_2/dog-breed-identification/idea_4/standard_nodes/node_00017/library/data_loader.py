import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import library.config as config


class DogDataset(Dataset):
    """
    Custom Dataset for loading Dog images based on metadata DataFrames.
    """

    def __init__(self, df, transform=None, class_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, breed).
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required if is_test is False.
            is_test (bool): Flag to indicate if this is the test set (no labels).
        """
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test
        self.root_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image and convert to RGB (standard for pre-trained models)
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError) as e:
            # Fallback or error handling could go here, but for this task we assume data integrity
            # based on the metadata check.
            raise e

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # For test set, return image and ID (for submission file)
            return image, row["id"]
        else:
            # For train/val, return image and label index
            label_str = row["breed"]
            label_idx = self.class_to_idx[label_str]
            return image, label_idx


def get_transforms(view_type):
    """
    Generates the transformation pipeline based on the view type configuration.

    Args:
        view_type (str): One of 'standard', 'global', 'local'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if view_type not in config.SCALE_CONFIGS:
        raise ValueError(
            f"Unknown view_type: {view_type}. Available: {list(config.SCALE_CONFIGS.keys())}"
        )

    scale_cfg = config.SCALE_CONFIGS[view_type]
    resize_param = scale_cfg["resize"]
    crop_param = scale_cfg["crop_size"]

    ops = []

    # 1. Resize
    # If resize_param is an int, it resizes the smaller edge to that size.
    # If it is a tuple, it resizes strictly to (h, w).
    ops.append(
        transforms.Resize(
            resize_param, interpolation=transforms.InterpolationMode.BICUBIC
        )
    )

    # 2. Center Crop (if applicable)
    if crop_param is not None:
        ops.append(transforms.CenterCrop(crop_param))

    # 3. ToTensor and Normalize
    ops.append(transforms.ToTensor())

    # Standard ImageNet normalization
    ops.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return transforms.Compose(ops)


def create_loaders(view_type, batch_size=config.BATCH_SIZE):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        view_type (str): The view configuration to use ('standard', 'global', 'local').
        batch_size (int): Batch size for the loaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
               classes is a sorted list of breed names corresponding to indices.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Handle Debug Mode
    if config.DEBUG:
        train_df = train_df.head(config.DEBUG_SAMPLES)
        val_df = val_df.head(config.DEBUG_SAMPLES)
        test_df = test_df.head(config.DEBUG_SAMPLES)

    # 3. Define Class Mapping (Alphabetical Sort)
    # We use the training set to determine classes.
    # Stratified split ensures all classes are in train, but we can verify.
    unique_breeds = sorted(train_df["breed"].unique())
    class_to_idx = {breed: i for i, breed in enumerate(unique_breeds)}

    # 4. Get Transforms
    transform = get_transforms(view_type)

    # 5. Create Datasets
    train_dataset = DogDataset(
        train_df, transform=transform, class_to_idx=class_to_idx, is_test=False
    )

    val_dataset = DogDataset(
        val_df, transform=transform, class_to_idx=class_to_idx, is_test=False
    )

    test_dataset = DogDataset(
        test_df, transform=transform, class_to_idx=None, is_test=True
    )

    # 6. Create DataLoaders
    # Num workers and pin_memory for efficiency
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, unique_breeds
