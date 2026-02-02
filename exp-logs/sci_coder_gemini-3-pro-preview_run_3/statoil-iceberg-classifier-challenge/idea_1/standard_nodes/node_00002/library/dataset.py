import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.model import load_and_process_data

# Set seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    Handles flattened image vectors and incidence angles.
    """

    def __init__(self, X, angles, y=None, ids=None):
        # Convert to FloatTensor
        self.X = torch.FloatTensor(X)
        self.angles = torch.FloatTensor(angles)

        # Handle targets: (N,) -> (N, 1)
        if y is not None:
            self.y = torch.FloatTensor(y).unsqueeze(1)
        else:
            self.y = None

        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return format expected by the model: (x_img, x_angle) and optionally label
        if self.y is not None:
            return self.X[idx], self.angles[idx], self.y[idx]
        else:
            return self.X[idx], self.angles[idx]


def get_data(
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Orchestrates data loading, preprocessing (scaling angles), and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to load pre-processed numpy arrays from cache.
        batch_size (int): Batch size for DataLoaders.
        debug_sample_size (int or None): If set, truncates train/val sets for debugging.

    Returns:
        dict: Contains 'train_loader', 'val_loader', 'test_loader', and 'test_ids'.
    """

    # 1. Load base data using library function
    # This handles: Metadata reading, JSON loading, Flattening, Imputing Angles, Scaling Images, Caching
    data_dict = load_and_process_data(load_cached_data=load_cached_data)

    X_train = data_dict["X_train"]
    angle_train = data_dict["angle_train"]
    y_train = data_dict["y_train"]

    X_val = data_dict["X_val"]
    angle_val = data_dict["angle_val"]
    y_val = data_dict["y_val"]

    X_test = data_dict["X_test"]
    angle_test = data_dict["angle_test"]
    test_ids = data_dict["test_ids"]

    # 2. Scale Incidence Angles
    # The library function imputes angles but does not scale them.
    # We apply StandardScaler here as required.
    angle_scaler = StandardScaler()

    # Reshape to (N, 1) for scaler, then flatten back to (N,)
    angle_train = angle_scaler.fit_transform(angle_train.reshape(-1, 1)).flatten()
    angle_val = angle_scaler.transform(angle_val.reshape(-1, 1)).flatten()
    angle_test = angle_scaler.transform(angle_test.reshape(-1, 1)).flatten()

    # 3. Debug Subsampling
    if debug_sample_size is not None:
        X_train = X_train[:debug_sample_size]
        angle_train = angle_train[:debug_sample_size]
        y_train = y_train[:debug_sample_size]

        X_val = X_val[:debug_sample_size]
        angle_val = angle_val[:debug_sample_size]
        y_val = y_val[:debug_sample_size]
        # We generally do not subsample test set to ensure submission file has all IDs,
        # unless strictly necessary for pipeline debugging.

    # 4. Create Datasets
    train_ds = IcebergDataset(X_train, angle_train, y_train)
    val_ds = IcebergDataset(X_val, angle_val, y_val)
    test_ds = IcebergDataset(X_test, angle_test, ids=test_ids)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "test_ids": test_ids,
    }
