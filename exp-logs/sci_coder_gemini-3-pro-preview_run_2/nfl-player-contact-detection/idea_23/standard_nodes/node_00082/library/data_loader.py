import os
import torch
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.feature_engineering import FeatureEngineer


class ContactDataset(Dataset):
    """
    PyTorch Dataset for KCVR-Net.
    Handles dual-stream input (Kinematic + Visual) and binary targets.
    """

    def __init__(self, X_kin, X_vis, y=None):
        self.X_kin = torch.FloatTensor(X_kin)
        self.X_vis = torch.FloatTensor(X_vis)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_kin[idx], self.X_vis[idx], self.y[idx]
        return self.X_kin[idx], self.X_vis[idx]


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Prepares DataLoaders for training and validation.
    Performs scaling (fit on train, transform val) and saves scalers.
    """
    # Initialize Feature Engineer
    fe = FeatureEngineer(use_tqdm=False)

    # Load Data
    # Note: FeatureEngineer handles caching internally
    X_kin_train, X_vis_train, y_train, _ = fe.process_data(split="train")
    X_kin_val, X_vis_val, y_val, _ = fe.process_data(split="validation")

    # Initialize Scalers
    scaler_kin = StandardScaler()
    scaler_vis = StandardScaler()

    # Fit on Train and Transform
    X_kin_train = scaler_kin.fit_transform(X_kin_train)
    X_vis_train = scaler_vis.fit_transform(X_vis_train)

    # Transform Validation
    X_kin_val = scaler_kin.transform(X_kin_val)
    X_vis_val = scaler_vis.transform(X_vis_val)

    # Save Scalers for Inference
    joblib.dump(scaler_kin, os.path.join(Config.WORKING_DIR, "scaler_kin.joblib"))
    joblib.dump(scaler_vis, os.path.join(Config.WORKING_DIR, "scaler_vis.joblib"))

    # Create Datasets
    train_ds = ContactDataset(X_kin_train, X_vis_train, y_train)
    val_ds = ContactDataset(X_kin_val, X_vis_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Prepares DataLoader for testing/inference.
    Loads saved scalers to ensure consistent normalization.
    """
    fe = FeatureEngineer(use_tqdm=False)

    # Load Test Data
    X_kin_test, X_vis_test, _, ids = fe.process_data(split="test")

    # Load Scalers
    scaler_kin_path = os.path.join(Config.WORKING_DIR, "scaler_kin.joblib")
    scaler_vis_path = os.path.join(Config.WORKING_DIR, "scaler_vis.joblib")

    if not os.path.exists(scaler_kin_path) or not os.path.exists(scaler_vis_path):
        raise FileNotFoundError(
            "Scalers not found. Run get_dataloaders (training) first."
        )

    scaler_kin = joblib.load(scaler_kin_path)
    scaler_vis = joblib.load(scaler_vis_path)

    # Transform
    X_kin_test = scaler_kin.transform(X_kin_test)
    X_vis_test = scaler_vis.transform(X_vis_test)

    # Create Dataset (No targets for inference)
    test_ds = ContactDataset(X_kin_test, X_vis_test, y=None)

    # Create DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, ids
