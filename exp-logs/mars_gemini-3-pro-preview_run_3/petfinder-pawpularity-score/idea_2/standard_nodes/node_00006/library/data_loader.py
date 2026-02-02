import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library import config, utils


class PetDataset(Dataset):
    """
    Custom Dataset for Pet Pawpularity Prediction.
    Loads images, processes them, and retrieves metadata features and targets.
    """

    def __init__(self, csv_path, mode="train", transform=None, debug=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Determines if targets are returned.
            transform (callable, optional): Optional transform to be applied on a sample.
            debug (bool): If True, use a small subset of the data.
        """
        self.mode = mode
        self.transform = transform
        self.df = pd.read_csv(csv_path)

        # Handle Debug mode
        if debug:
            self.df = self.df.head(100).reset_index(drop=True)

        # Pre-compute full file paths
        # metadata file_path is relative to input dir, e.g., "train/id.jpg"
        self.df["full_path"] = self.df["file_path"].apply(
            lambda x: os.path.join(config.INPUT_DIR, x)
        )

        # Extract Metadata Features
        self.meta_features = self.df[config.META_FEATURES].values.astype(np.float32)

        # Extract Targets if not in test mode
        if self.mode != "test":
            self.targets = self.df["Pawpularity"].values.astype(np.float32)
        else:
            # Dummy targets for test set to maintain consistent signature
            self.targets = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.df.iloc[idx]["full_path"]

        # Use OpenCV to read image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though metadata generation ensures existence)
            # Create a black image of correct size
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            image = self.transform(image)

        # 3. Get Metadata and Target
        metadata = torch.tensor(self.meta_features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Return tuple: (image_tensor, metadata_vector, target_scalar)
        return image, metadata, target


def get_transforms(img_size=config.IMG_SIZE):
    """
    Returns the transformation pipeline for the images.

    Args:
        img_size (int): Target size for resizing.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts [0, 255] to [0.0, 1.0]
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, debug=config.DEBUG
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform = get_transforms()

    # Initialize Datasets
    train_dataset = PetDataset(
        csv_path=config.TRAIN_META_PATH, mode="train", transform=transform, debug=debug
    )

    val_dataset = PetDataset(
        csv_path=config.VAL_META_PATH, mode="val", transform=transform, debug=debug
    )

    test_dataset = PetDataset(
        csv_path=config.TEST_META_PATH, mode="test", transform=transform, debug=debug
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # We shuffle in feature extraction if needed, but for extraction we just need linear scan usually.
        # However, standard practice for training is shuffle=True.
        # Since we are caching features first, order doesn't strictly matter
        # as long as features and targets align. We'll keep shuffle=False
        # to ensure deterministic feature matrix construction order.
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

    return train_loader, val_loader, test_loader
