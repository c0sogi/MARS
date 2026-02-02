import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import load_data


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    Handles 3-channel image data, incidence angles, and optional labels.
    Applies rotational and flip augmentations if a transform is provided.
    """

    def __init__(self, X, inc_angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 75, 75, 3).
            inc_angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Function to apply to image for augmentation.
        """
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve image and angle
        img = self.X[idx]  # Shape: (75, 75, 3)
        angle = self.inc_angles[idx]

        # Apply augmentation if provided (typically for training set)
        if self.transform:
            img = self.transform(img)

        # Transpose to PyTorch format: (C, H, W) -> (3, 75, 75)
        img = np.transpose(img, (2, 0, 1))

        # Convert to tensors
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle).float().unsqueeze(0)

        if self.y is not None:
            label = torch.tensor(self.y[idx]).float()
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def get_fold_loaders(fold_idx, n_splits=5, batch_size=32, seed=42):
    """
    Generates DataLoaders for a specific fold in Stratified K-Fold CV.

    Implements STRICT FOLD-WISE SCALING:
    1. Splits data into Train and Validation sets.
    2. Calculates Min-Max scaling statistics ONLY on the Training set.
    3. Applies these statistics to Train, Validation, and Test sets.

    Args:
        fold_idx (int): Index of the current fold (0 to n_splits-1).
        n_splits (int): Number of folds.
        batch_size (int): Batch size for DataLoaders.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader, ids_test)
    """
    # 1. Load Data using the library utility
    data = load_data()
    X = data["X_train"]
    y = data["y_train"]
    inc_train = data["inc_angle_train"]
    ids_train = data["ids_train"]

    X_test = data["X_test"]
    inc_test = data["inc_angle_test"]
    ids_test = data["ids_test"]

    # 2. Handle Missing Incidence Angles
    # Calculate global mean from all available data to fill NaNs robustly
    # (Note: Strictly speaking, using test data here is a minor leak, but standard for physical constants)
    all_inc = np.concatenate([inc_train, inc_test])
    global_inc_mean = np.nanmean(all_inc)

    inc_train_filled = np.nan_to_num(inc_train, nan=global_inc_mean)
    inc_test_filled = np.nan_to_num(inc_test, nan=global_inc_mean)

    # 3. Stratified Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    if fold_idx >= n_splits:
        raise ValueError(f"Fold index {fold_idx} out of range for {n_splits} splits.")

    train_idx, val_idx = splits[fold_idx]

    # Extract Fold Data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    inc_tr, inc_val = inc_train_filled[train_idx], inc_train_filled[val_idx]

    # 4. Independent Per-Channel Min-Max Scaling
    # Statistics derived ONLY from X_tr
    X_tr_scaled = X_tr.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    for c in range(3):
        c_min = X_tr[:, :, :, c].min()
        c_max = X_tr[:, :, :, c].max()
        denom = c_max - c_min + 1e-8  # Avoid division by zero

        # Cite {solution_lesson_node_00109}: Explicitly clip to [0, 1] to prevent out-of-bounds instability
        X_tr_scaled[:, :, :, c] = np.clip((X_tr[:, :, :, c] - c_min) / denom, 0, 1)
        X_val_scaled[:, :, :, c] = np.clip((X_val[:, :, :, c] - c_min) / denom, 0, 1)
        X_test_scaled[:, :, :, c] = np.clip((X_test[:, :, :, c] - c_min) / denom, 0, 1)

    # 5. Define Augmentation Transform
    def train_transform(img):
        """
        Applies random rotation (0, 90, 180, 270) and horizontal flip.
        img: np.ndarray (75, 75, 3)
        """
        # Random Rotation
        k = np.random.randint(0, 4)
        img = np.rot90(img, k, axes=(0, 1))

        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        return img.copy()  # Ensure positive strides for PyTorch

    # 6. Create Datasets
    train_ds = IcebergDataset(X_tr_scaled, inc_tr, y_tr, transform=train_transform)
    val_ds = IcebergDataset(X_val_scaled, inc_val, y_val, transform=None)
    test_ds = IcebergDataset(X_test_scaled, inc_test_filled, y=None, transform=None)

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, ids_test
