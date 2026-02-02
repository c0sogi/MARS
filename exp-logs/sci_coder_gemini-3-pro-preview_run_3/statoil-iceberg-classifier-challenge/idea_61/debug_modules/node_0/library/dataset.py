import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    Handles 3-channel image construction, angle imputation, and augmentation.
    """

    def __init__(
        self, images, angles, labels=None, ids=None, transform=None, impute_angle=0.0
    ):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75).
            angles (np.ndarray): Shape (N,). Contains NaNs.
            labels (np.ndarray, optional): Shape (N,). Binary targets.
            ids (np.ndarray, optional): Shape (N,). Image IDs.
            transform (callable, optional): PyTorch transforms.
            impute_angle (float): Value to replace NaNs in angles.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.impute_angle = impute_angle

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image: (3, 75, 75)
        img = self.images[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        # Apply augmentations (e.g., flips)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Retrieve and impute angle
        angle = self.angles[idx]
        if np.isnan(angle):
            angle = self.impute_angle

        # Angle as tensor (scalar)
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        # Return (image, angle, label) or (image, angle, id)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            img_id = self.ids[idx]
            return img_tensor, angle_tensor, img_id


def _process_json_data(json_path, is_train=True):
    """
    Reads JSON, reshapes bands, computes 3rd channel, extracts metadata.
    Returns: X (images), angles, ids, y (labels if train)
    """
    # Use pandas for efficient JSON reading
    df = pd.read_json(json_path)

    # Process Images
    # Convert lists to numpy arrays and reshape
    # df['band_1'] is a Series of lists. .tolist() makes it a list of lists.
    # np.array converts to (N, 5625). Reshape to (N, 75, 75).
    b1 = np.array(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Compute Band 3 (Average of HH and HV)
    b3 = (b1 + b2) / 2.0

    # Stack into (N, 3, 75, 75)
    # Axis 1 is channel dimension
    X = np.stack([b1, b2, b3], axis=1)

    # Process Angles
    # Coerce 'na' to NaN
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # IDs
    ids = df["id"].values

    y = None
    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)

    return X, angles, ids, y


def _get_cached_arrays(is_train=True, load_cached_data=True):
    """
    Handles caching logic. Loads from .npy if available/requested, else processes JSON.
    """
    suffix = "train" if is_train else "test"
    cache_dir = Config.CACHE_DIR

    # Define file paths
    path_X = os.path.join(cache_dir, f"X_{suffix}.npy")
    path_ang = os.path.join(cache_dir, f"angles_{suffix}.npy")
    path_ids = os.path.join(cache_dir, f"ids_{suffix}.npy")
    path_y = os.path.join(cache_dir, f"y_{suffix}.npy")

    files_exist = (
        os.path.exists(path_X) and os.path.exists(path_ang) and os.path.exists(path_ids)
    )
    if is_train:
        files_exist = files_exist and os.path.exists(path_y)

    if load_cached_data and files_exist:
        X = np.load(path_X)
        angles = np.load(path_ang)
        ids = np.load(path_ids)
        y = np.load(path_y) if is_train else None
        return X, angles, ids, y

    # Process from scratch
    json_path = Config.TRAIN_JSON if is_train else Config.TEST_JSON
    X, angles, ids, y = _process_json_data(json_path, is_train)

    # Save to cache
    np.save(path_X, X)
    np.save(path_ang, angles)
    np.save(path_ids, ids)
    if is_train:
        np.save(path_y, y)

    return X, angles, ids, y


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Uses metadata files to split the training data.
    Computes angle imputation value solely from the training split.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Get Full Data Arrays (Cached or Processed)
    X_train_full, ang_train_full, ids_train_full, y_train_full = _get_cached_arrays(
        is_train=True, load_cached_data=load_cached_data
    )
    X_test, ang_test, ids_test, _ = _get_cached_arrays(
        is_train=False, load_cached_data=load_cached_data
    )

    # 2. Load Metadata for Splits
    # These files contain 'original_index' which maps back to the arrays above
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)

    train_indices = train_meta["original_index"].values
    val_indices = val_meta["original_index"].values

    # 3. Compute Imputation Value (Median Angle) from TRAIN split only
    # Filter out NaNs
    train_angles_subset = ang_train_full[train_indices]
    valid_angles = train_angles_subset[~np.isnan(train_angles_subset)]

    if len(valid_angles) > 0:
        impute_val = float(np.median(valid_angles))
    else:
        impute_val = 0.0

    # 4. Define Transforms
    # Only augment training data
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        images=X_train_full[train_indices],
        angles=ang_train_full[train_indices],
        labels=y_train_full[train_indices],
        ids=ids_train_full[train_indices],
        transform=train_transform,
        impute_angle=impute_val,
    )

    val_dataset = IcebergDataset(
        images=X_train_full[val_indices],
        angles=ang_train_full[val_indices],
        labels=y_train_full[val_indices],
        ids=ids_train_full[val_indices],
        transform=None,  # No augmentation for validation
        impute_angle=impute_val,
    )

    test_dataset = IcebergDataset(
        images=X_test,
        angles=ang_test,
        labels=None,
        ids=ids_test,
        transform=None,
        impute_angle=impute_val,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize batchnorm statistics
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

    return train_loader, val_loader, test_loader
