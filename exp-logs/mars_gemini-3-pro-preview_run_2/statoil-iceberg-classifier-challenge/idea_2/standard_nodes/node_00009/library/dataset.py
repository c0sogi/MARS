import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from library.utils import load_dataset
from library.config import BATCH_SIZE, DEBUG, DEBUG_SIZE


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg Classifier.
    Wraps preprocessed numpy arrays for images, angles, and labels.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32.
            ids (np.ndarray, optional): Shape (N,), string/object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors
        # Images are already (C, H, W) from utils
        image = torch.from_numpy(self.images[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "angle": angle}

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = str(self.ids[idx])

        return sample


def get_dataloaders(
    load_cached_data=True, batch_size=BATCH_SIZE, debug=DEBUG, debug_size=DEBUG_SIZE
):
    """
    Loads the dataset using the utility library and returns PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from disk cache.
        batch_size (int): Batch size for the DataLoaders.
        debug (bool): If True, truncates datasets to debug_size.
        debug_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load all data (cached or processed from scratch via utils)
    data = load_dataset(load_cached_data=load_cached_data)

    X_train = data["X_train"]
    angle_train = data["angle_train"]
    y_train = data["y_train"]
    id_train = data["id_train"]

    X_val = data["X_val"]
    angle_val = data["angle_val"]
    y_val = data["y_val"]
    id_val = data["id_val"]

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    id_test = data["id_test"]

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled. Truncating datasets to {debug_size} samples.")
        X_train = X_train[:debug_size]
        angle_train = angle_train[:debug_size]
        y_train = y_train[:debug_size]
        id_train = id_train[:debug_size]

        X_val = X_val[:debug_size]
        angle_val = angle_val[:debug_size]
        y_val = y_val[:debug_size]
        id_val = id_val[:debug_size]

        X_test = X_test[:debug_size]
        angle_test = angle_test[:debug_size]
        id_test = id_test[:debug_size]

    # Define Augmentation for Training
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    # Instantiate Datasets
    train_ds = IcebergDataset(
        X_train, angle_train, y_train, id_train, transform=train_transform
    )
    val_ds = IcebergDataset(X_val, angle_val, y_val, id_val)
    test_ds = IcebergDataset(X_test, angle_test, labels=None, ids=id_test)

    # Create DataLoaders
    # Pin memory helps with faster transfer to GPU
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
