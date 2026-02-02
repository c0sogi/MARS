import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import random

# Import from provided libraries
from library.config import Config
from library.feature_extractor import extract_features, extract_spectrograms


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class VolcanoDataset(Dataset):
    """
    Custom Dataset for Volcano Seismic Data (Hybrid).
    Wraps Spectrograms, Tabular Features, and Targets.
    Applies SpecAugment (Cite solution_lesson_node_00007).
    """

    def __init__(self, specs, tabular, targets=None, augment=False):
        self.specs = torch.tensor(specs, dtype=torch.float32)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.augment = augment

        # SpecAugment Transforms
        if self.augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        tab = self.tabular[idx]

        if self.augment:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        if self.targets is not None:
            return (spec, tab), self.targets[idx]
        return (spec, tab)


def prepare_data(
    debug_size=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Tabular Features
    print("Preparing Tabular Data...")
    df_train = extract_features(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_FEATURES_CACHE,
        load_cached_data,
        debug_size,
    )
    df_val = extract_features(
        Config.VAL_METADATA_PATH,
        Config.VAL_FEATURES_CACHE,
        load_cached_data,
        debug_size,
    )

    # 2. Load Spectrograms
    print("Preparing Spectrograms...")
    specs_train = extract_spectrograms(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_SPEC_CACHE,
        load_cached_data,
        debug_size,
    )
    specs_val = extract_spectrograms(
        Config.VAL_METADATA_PATH, Config.VAL_SPEC_CACHE, load_cached_data, debug_size
    )

    # 3. Process Tabular
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["time_to_eruption"].values.astype(np.float32)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["time_to_eruption"].values.astype(np.float32)

    input_dim = len(feature_cols)

    # 4. Scaling Tabular
    print("Fitting Feature Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Scaling Spectrograms (Global Mean/Std)
    spec_mean = np.mean(specs_train)
    spec_std = np.std(specs_train)
    specs_train_scaled = (specs_train - spec_mean) / (spec_std + 1e-6)
    specs_val_scaled = (specs_val - spec_mean) / (spec_std + 1e-6)

    # 6. Scaling Targets
    print("Fitting Target Scaler...")
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1.0

    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    # 7. Save Parameters
    np.save(Config.SCALER_MEAN_PATH, scaler.mean_)
    np.save(Config.SCALER_SCALE_PATH, scaler.scale_)
    np.save(Config.TARGET_MEAN_PATH, y_mean)
    np.save(Config.TARGET_STD_PATH, y_std)
    np.save(Config.SPEC_MEAN_PATH, spec_mean)
    np.save(Config.SPEC_STD_PATH, spec_std)

    # 8. Create Datasets
    # Apply SpecAugment only to Training set
    train_dataset = VolcanoDataset(
        specs_train_scaled, X_train_scaled, y_train_scaled, augment=True
    )
    val_dataset = VolcanoDataset(
        specs_val_scaled, X_val_scaled, y_val_scaled, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, scaler, input_dim
