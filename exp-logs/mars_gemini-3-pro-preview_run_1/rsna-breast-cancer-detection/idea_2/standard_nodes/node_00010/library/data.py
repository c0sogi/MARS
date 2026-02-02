import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_and_process_image


def prepare_metadata(load_cached_data=True):
    """
    Loads metadata, handles missing values, computes scaling stats for age,
    and caches the processed dataframes.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "processed_train.parquet")
    val_cache = os.path.join(cache_dir, "processed_val.parquet")
    test_cache = os.path.join(cache_dir, "processed_test.parquet")
    stats_cache = os.path.join(cache_dir, "age_stats.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(stats_cache)
        ):
            try:
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)
                stats = np.load(stats_cache, allow_pickle=True).item()
                return df_train, df_val, df_test, stats
            except Exception:
                pass  # Fallback to re-compute

    # 2. Load Raw Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # 3. Handle Age (Imputation + Stats Calculation)
    # Calculate stats on TRAIN only to avoid leakage
    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()

    # Fill missing age with mean
    df_train["age"] = df_train["age"].fillna(age_mean)
    df_val["age"] = df_val["age"].fillna(age_mean)
    df_test["age"] = df_test["age"].fillna(age_mean)

    # Store stats
    stats = {"age_mean": age_mean, "age_std": age_std}

    # 4. Handle Density (Auxiliary Target)
    # Map density to integers. Missing density becomes -1.
    def map_density(val):
        return Config.DENSITY_MAP.get(val, -1)

    if "density" in df_train.columns:
        df_train["density_label"] = df_train["density"].apply(map_density)
    else:
        df_train["density_label"] = -1

    if "density" in df_val.columns:
        df_val["density_label"] = df_val["density"].apply(map_density)
    else:
        df_val["density_label"] = -1

    # Test set doesn't have density, set to -1
    df_test["density_label"] = -1

    # 5. Handle Implant (Ensure binary)
    # Fill NaNs with 0 (assuming no implant if unknown)
    for df in [df_train, df_val, df_test]:
        if "implant" in df.columns:
            df["implant"] = df["implant"].fillna(0).astype(int)
        else:
            df["implant"] = 0

    # 6. Save to Cache
    try:
        df_train.to_parquet(train_cache)
        df_val.to_parquet(val_cache)
        df_test.to_parquet(test_cache)
        np.save(stats_cache, stats)
    except Exception:
        pass

    return df_train, df_val, df_test, stats


class MammogramDataset(Dataset):
    def __init__(self, df, age_stats, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            age_stats (dict): Dictionary containing 'age_mean' and 'age_std'.
            transforms (albumentations.Compose): Image augmentations.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.age_mean = age_stats["age_mean"]
        self.age_std = age_stats["age_std"]
        self.transforms = transforms
        self.mode = mode

        # Pre-extract columns to avoid overhead in __getitem__
        self.file_paths = df["file_path"].values
        self.ages = df["age"].values
        self.implants = df["implant"].values

        # Targets
        if self.mode != "test":
            self.cancer_labels = df["cancer"].values.astype(np.float32)
            self.density_labels = df["density_label"].values.astype(np.int64)
        else:
            self.cancer_labels = np.zeros(len(df), dtype=np.float32)
            self.density_labels = np.full(len(df), -1, dtype=np.int64)

        # For submission mapping
        if "prediction_id" in df.columns:
            self.prediction_ids = df["prediction_id"].values
        else:
            self.prediction_ids = np.array([""] * len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        # Returns (H, W) uint8
        img = load_and_process_image(self.file_paths[idx], load_cached_data=True)

        # 2. Apply Augmentations (Image Only)
        if self.transforms:
            # Albumentations expects HWC or HW
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # 3. Normalize Image to [0, 1]
        img = img.astype(np.float32) / 255.0

        # 4. Prepare Metadata Channels
        h, w = img.shape

        # Age Channel: Standard Scaled -> Broadcast
        age_val = self.ages[idx]
        age_norm = (age_val - self.age_mean) / (self.age_std + 1e-6)
        age_channel = np.full((h, w), age_norm, dtype=np.float32)

        # Implant Channel: Binary -> Broadcast
        implant_val = self.implants[idx]
        implant_channel = np.full((h, w), implant_val, dtype=np.float32)

        # 5. Stack Channels (3, H, W)
        # Channel 0: Image
        # Channel 1: Age
        # Channel 2: Implant
        input_tensor = np.stack([img, age_channel, implant_channel], axis=0)
        input_tensor = torch.from_numpy(input_tensor)

        # 6. Prepare Targets
        cancer_target = torch.tensor(self.cancer_labels[idx], dtype=torch.float)
        density_target = torch.tensor(self.density_labels[idx], dtype=torch.long)

        if self.mode == "test":
            return input_tensor, self.prediction_ids[idx]
        else:
            return input_tensor, cancer_target, density_target


def get_transforms(mode="train"):
    """
    Returns albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.2
                ),
                # Note: Resize is handled in load_and_process_image
            ]
        )
    else:
        return None


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares datasets and returns dataloaders for train, val, and test.
    """
    # 1. Prepare Dataframes
    df_train, df_val, df_test, stats = prepare_metadata(
        load_cached_data=load_cached_data
    )

    if debug:
        df_train = df_train.sample(n=100, random_state=Config.SEED).reset_index(
            drop=True
        )
        df_val = df_val.sample(n=50, random_state=Config.SEED).reset_index(drop=True)
        df_test = df_test.sample(n=50, random_state=Config.SEED).reset_index(drop=True)

    # 2. Create Datasets
    train_dataset = MammogramDataset(
        df_train, stats, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = MammogramDataset(
        df_val, stats, transforms=get_transforms("val"), mode="val"
    )

    test_dataset = MammogramDataset(
        df_test, stats, transforms=get_transforms("test"), mode="test"
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
