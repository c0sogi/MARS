import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import load_dataset


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    Wraps numpy arrays and handles conversion to PyTorch tensors and augmentation.
    """

    def __init__(self, X, angles, labels=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Target labels of shape (N,). Defaults to None.
            ids (np.ndarray, optional): Sample IDs of shape (N,). Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to PyTorch tensors
        # img is already float32 from load_dataset, safe to convert
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply Augmentations (only affects image tensor)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return tuple based on available data (Test vs Train/Val)
        if self.ids is not None:
            # Inference Mode: Return ID for submission generation
            sample_id = self.ids[idx]
            return img_tensor, angle_tensor, sample_id
        else:
            # Training/Validation Mode: Return Label
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # 1. Load Data using library utility
    # load_dataset handles caching and metadata reading
    X_train, angles_train, y_train = load_dataset("train")
    X_val, angles_val, y_val = load_dataset("val")
    X_test, angles_test, ids_test = load_dataset("test")

    # 2. Handle Debug Mode (Subset Data)
    if debug:
        print("Debug mode enabled: Slicing datasets to 32 samples.")
        subset_size = 32
        X_train = X_train[:subset_size]
        angles_train = angles_train[:subset_size]
        y_train = y_train[:subset_size]

        X_val = X_val[:subset_size]
        angles_val = angles_val[:subset_size]
        y_val = y_val[:subset_size]

        X_test = X_test[:subset_size]
        angles_test = angles_test[:subset_size]
        ids_test = ids_test[:subset_size]

    # 3. Define Transforms
    # We use random flips for training. Radar data is invariant to horizontal/vertical flips.
    # Input is (3, 75, 75), transforms work directly on tensors.
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No augmentation for validation or test
    val_transform = None
    test_transform = None

    # 4. Create Dataset Instances
    train_dataset = IcebergDataset(
        X_train, angles_train, labels=y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, angles_val, labels=y_val, transform=val_transform
    )
    test_dataset = IcebergDataset(
        X_test, angles_test, ids=ids_test, transform=test_transform
    )

    # 5. Create DataLoaders
    # pin_memory=True speeds up host-to-device transfer for CUDA
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
