import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.feature_engineering import generate_dataset


class ContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Handles dual-stream input: Kinematic features and Visual features.
    """

    def __init__(self, df, feature_cols, visual_cols, target_col=None):
        # Convert to float32 for PyTorch
        self.features = df[feature_cols].values.astype(np.float32)
        self.visuals = df[visual_cols].values.astype(np.float32)

        # Handle target if present (Train/Val)
        if target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x_kin = self.features[idx]
        x_vis = self.visuals[idx]

        if self.targets is not None:
            y = self.targets[idx]
            return x_kin, x_vis, y
        return x_kin, x_vis


def prepare_dataloaders(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Orchestrates data loading, splitting, scaling, and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders.
        (kin_dim, vis_dim): Input dimensions for the model.
    """
    # 1. Load Data using Feature Engineering Library
    # This handles the heavy lifting of feature generation and caching
    print("Loading datasets...")
    df_train_all = generate_dataset(mode="train", load_cached_data=load_cached_data)
    df_test = generate_dataset(mode="test", load_cached_data=load_cached_data)

    # 2. Split Train/Val based on Metadata
    # We strictly respect the game_play split defined in metadata/validation.csv
    print("Splitting training and validation sets...")
    meta_val = pd.read_csv(Config.METADATA_VAL)
    val_gps = meta_val["game_play"].unique()

    is_val = df_train_all["game_play"].isin(val_gps)
    df_train = df_train_all[~is_val].copy()
    df_val = df_train_all[is_val].copy()

    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")

    # 3. Identify Feature Columns
    # Exclude metadata columns to isolate features
    all_cols = df_train.columns

    # Visual columns start with 'v_' based on feature_engineering.py logic
    vis_cols = [c for c in all_cols if c.startswith("v_")]

    # Exclude IDs and Target from Kinematic columns
    exclude_cols = [
        "contact_id",
        "game_play",
        "contact",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ] + vis_cols

    kin_cols = [c for c in all_cols if c not in exclude_cols]

    print(f"Kinematic Features ({len(kin_cols)}): {kin_cols[:5]}...")
    print(f"Visual Features ({len(vis_cols)}): {vis_cols}")

    # 4. Scale Data
    # Fit scaler ONLY on training data to prevent leakage
    print("Scaling features...")
    scaler = StandardScaler()
    df_train[kin_cols] = scaler.fit_transform(df_train[kin_cols])

    # Transform Val and Test using Train statistics
    df_val[kin_cols] = scaler.transform(df_val[kin_cols])
    df_test[kin_cols] = scaler.transform(df_test[kin_cols])

    # Save scaler for potential inference usage later
    joblib.dump(scaler, Config.SCALER_PATH)

    # 5. Create Datasets
    train_ds = ContactDataset(df_train, kin_cols, vis_cols, target_col="contact")
    val_ds = ContactDataset(df_val, kin_cols, vis_cols, target_col="contact")
    test_ds = ContactDataset(df_test, kin_cols, vis_cols, target_col=None)

    # 6. Create DataLoaders
    # Use num_workers=4 as suggested in Config examples
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, (len(kin_cols), len(vis_cols))
