import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler

from library.config import BATCH_SIZE, NUM_WORKERS, SCALER_PATH, CACHE_DIR, SEED
from library.preprocessing import load_data

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class GNSSWindowDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Stream SCR-CNN.
    Provides:
        - Kinematic Sequence: (Window_Size, Features)
        - Sky Context Vector: (Features,)
        - Target Residuals: (2,) [dLat_m, dLon_m]
    """

    def __init__(self, X_kin, X_sky, y=None):
        """
        Args:
            X_kin (np.ndarray): Kinematic features (N, Window, Feat)
            X_sky (np.ndarray): Sky context features (N, Feat)
            y (np.ndarray, optional): Target residuals (N, 2). None for test.
        """
        self.X_kin = torch.FloatTensor(X_kin)
        self.X_sky = torch.FloatTensor(X_sky)

        if y is not None:
            self.y = torch.FloatTensor(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        kinematic = self.X_kin[idx]
        sky = self.X_sky[idx]

        # Kinematic stream expects (Channels, Sequence_Length) for 1D CNN usually,
        # but PyTorch Conv1d takes (Batch, Channels, Length).
        # Here we return (Length, Channels) or (Channels, Length).
        # Standard PyTorch Conv1d expects input (N, C, L).
        # Our data is (N, L, C). We should transpose it here to (C, L).
        kinematic = kinematic.transpose(0, 1)

        if self.y is not None:
            target = self.y[idx]
            return kinematic, sky, target
        else:
            return kinematic, sky


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Loads data, fits/loads scalers, and returns DataLoaders.

    Args:
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker subprocesses.
        load_cached_data (bool): Whether to use cached preprocessed data and scalers.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_meta)
    """
    # 1. Load Raw/Cached Numpy Arrays
    # This uses the preprocessing library which handles the heavy lifting of windowing
    (train_data, val_data, test_data) = load_data(load_cached_data=load_cached_data)

    train_X_kin, train_X_sky, train_y, _ = train_data
    val_X_kin, val_X_sky, val_y, _ = val_data
    test_X_kin, test_X_sky, _, test_meta = test_data

    # 2. Handle Scaling
    # We need two scalers: one for kinematic features, one for sky features.
    # We flatten the kinematic features to fit the scaler, then reshape back.

    scaler_kin_path = os.path.join(CACHE_DIR, "scaler_kin.joblib")
    scaler_sky_path = os.path.join(CACHE_DIR, "scaler_sky.joblib")

    if (
        load_cached_data
        and os.path.exists(scaler_kin_path)
        and os.path.exists(scaler_sky_path)
    ):
        print("Loading scalers from cache...")
        scaler_kin = joblib.load(scaler_kin_path)
        scaler_sky = joblib.load(scaler_sky_path)
    else:
        print("Fitting scalers on training data...")
        # Kinematic: (N, L, C) -> (N*L, C)
        N, L, C = train_X_kin.shape
        scaler_kin = StandardScaler()
        scaler_kin.fit(train_X_kin.reshape(N * L, C))

        # Sky: (N, C)
        scaler_sky = StandardScaler()
        scaler_sky.fit(train_X_sky)

        # Save scalers
        os.makedirs(CACHE_DIR, exist_ok=True)
        joblib.dump(scaler_kin, scaler_kin_path)
        joblib.dump(scaler_sky, scaler_sky_path)
        print("Scalers saved.")

    # 3. Apply Scaling
    print("Applying scaling...")

    def scale_kin(X, scaler):
        if len(X) == 0:
            return X
        N, L, C = X.shape
        X_flat = X.reshape(N * L, C)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(N, L, C)

    def scale_sky(X, scaler):
        if len(X) == 0:
            return X
        return scaler.transform(X)

    train_X_kin = scale_kin(train_X_kin, scaler_kin)
    train_X_sky = scale_sky(train_X_sky, scaler_sky)

    val_X_kin = scale_kin(val_X_kin, scaler_kin)
    val_X_sky = scale_sky(val_X_sky, scaler_sky)

    test_X_kin = scale_kin(test_X_kin, scaler_kin)
    test_X_sky = scale_sky(test_X_sky, scaler_sky)

    # 4. Create Datasets
    train_dataset = GNSSWindowDataset(train_X_kin, train_X_sky, train_y)
    val_dataset = GNSSWindowDataset(val_X_kin, val_X_sky, val_y)
    test_dataset = GNSSWindowDataset(test_X_kin, test_X_sky, None)

    # 5. Create DataLoaders
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

    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, test_loader, test_meta
