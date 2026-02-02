import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import load_image, seed_everything


def process_metadata(load_cached_data=True):
    """
    Processes metadata to pair images with their contralateral views and computes age statistics.
    Implements caching using parquet files.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "processed_train.parquet")
    val_cache = os.path.join(cache_dir, "processed_val.parquet")
    test_cache = os.path.join(cache_dir, "processed_test.parquet")
    stats_cache = os.path.join(cache_dir, "age_stats.npy")

    # 1. Try Load Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(stats_cache)
        ):

            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)
            age_stats = np.load(stats_cache, allow_pickle=True).item()
            return df_train, df_val, df_test, age_stats

    # 2. Process from Scratch
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Calculate Age Stats (Mean/Std) from Train only
    # Fill missing age with mean for calculation
    train_ages = df_train[Config.AGE_COL].dropna()
    age_mean = train_ages.mean()
    age_std = train_ages.std()

    # Handle missing ages in dataframes (impute with mean)
    df_train[Config.AGE_COL] = df_train[Config.AGE_COL].fillna(age_mean)
    df_val[Config.AGE_COL] = df_val[Config.AGE_COL].fillna(age_mean)
    df_test[Config.AGE_COL] = df_test[Config.AGE_COL].fillna(age_mean)

    age_stats = {"mean": age_mean, "std": age_std}

    # Helper to find contralateral path
    def add_contralateral_path(df):
        # Create a lookup key: (patient_id, view, laterality) -> file_path
        # We need to find (patient_id, view, OPPOSITE_laterality)

        # Build lookup dict
        lookup = {}
        for idx, row in df.iterrows():
            key = (
                row[Config.PATIENT_ID_COL],
                row[Config.VIEW_COL],
                row[Config.LATERALITY_COL],
            )
            lookup[key] = row[Config.FILE_PATH_COL]

        contra_paths = []
        for idx, row in df.iterrows():
            pid = row[Config.PATIENT_ID_COL]
            view = row[Config.VIEW_COL]
            lat = row[Config.LATERALITY_COL]

            opp_lat = "R" if lat == "L" else "L"
            target_key = (pid, view, opp_lat)

            path = lookup.get(target_key, None)
            contra_paths.append(path)

        df["contra_file_path"] = contra_paths
        return df

    # We need to process train/val separately for pairing?
    # Actually, pairing is within patient. Patients are unique to splits.
    # So we can process each df independently.
    df_train = add_contralateral_path(df_train)
    df_val = add_contralateral_path(df_val)
    df_test = add_contralateral_path(df_test)

    # 3. Save Cache
    df_train.to_parquet(train_cache)
    df_val.to_parquet(val_cache)
    df_test.to_parquet(test_cache)
    np.save(stats_cache, age_stats)

    return df_train, df_val, df_test, age_stats


class PairedBreastCancerDataset(Dataset):
    def __init__(self, df, image_dir, age_stats, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.age_mean = age_stats["mean"]
        self.age_std = age_stats["std"]
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        # Construct full path. Dataframe has relative path.
        target_path = os.path.join(Config.INPUT_DIR, row[Config.FILE_PATH_COL])

        # Load image (H, W) normalized [0, 1]
        target_img = load_image(target_path)

        # 2. Load Contralateral Image
        contra_rel_path = row["contra_file_path"]
        if contra_rel_path is not None and isinstance(contra_rel_path, str):
            contra_path = os.path.join(Config.INPUT_DIR, contra_rel_path)
            if os.path.exists(contra_path):
                contra_img = load_image(contra_path)
            else:
                # Fallback if file missing despite metadata saying it exists (should fail loud usually,
                # but here we treat missing file same as missing breast for robustness if metadata is stale)
                contra_img = np.zeros_like(target_img)
        else:
            # Physically missing contralateral breast
            contra_img = np.zeros_like(target_img)

        # 3. Prepare Metadata Maps
        # Age Standardization
        age = row[Config.AGE_COL]
        age_norm = (age - self.age_mean) / (self.age_std + 1e-7)

        # Implant
        implant = row[Config.IMPLANT_COL] if Config.IMPLANT_COL in row else 0
        implant = float(implant)

        # 4. Construct 3-Channel Tensors (H, W, 3)
        # We construct numpy arrays first for Albumentations

        def create_3ch_tensor(img, age_val, implant_val):
            h, w = img.shape
            # Create constant maps
            age_map = np.full((h, w), age_val, dtype=np.float32)
            implant_map = np.full((h, w), implant_val, dtype=np.float32)
            # Stack: (H, W, 3)
            return np.stack([img, age_map, implant_map], axis=-1)

        target_tensor_np = create_3ch_tensor(target_img, age_norm, implant)

        # For Contralateral:
        # If image was found, use it. If zero (missing), use zero image.
        # CRITICAL: Age and Implant for Contra must match Target to ensure
        # (Age_target - Age_contra) == 0 in the difference module.
        contra_tensor_np = create_3ch_tensor(contra_img, age_norm, implant)

        # 5. Augmentation
        if self.transform:
            # Albumentations expects HWC. We have HWC.
            # We use 'image0' as the additional target for the contralateral image
            # to ensure identical geometric transforms.
            augmented = self.transform(image=target_tensor_np, image0=contra_tensor_np)
            target_tensor_np = augmented["image"]
            contra_tensor_np = augmented["image0"]

        # 6. Convert to PyTorch Tensor (CHW)
        # If ToTensorV2 was used in transform, it's already tensor CHW.
        # If not (e.g. custom), we need to convert.
        # Assuming get_transforms includes ToTensorV2.

        sample = {
            "image": target_tensor_np,  # (3, H, W)
            "contra_image": contra_tensor_np,  # (3, H, W)
        }

        if not self.is_test:
            sample["label"] = torch.tensor(row[Config.TARGET_COL], dtype=torch.float32)
        else:
            sample[Config.ID_COL] = row[Config.ID_COL]

        return sample


def get_transforms(data="train"):
    """
    Returns albumentations transforms.
    Synchronized geometric augmentation for paired images.
    """
    height, width = Config.IMG_SIZE

    if data == "train":
        return A.Compose(
            [
                A.Resize(height, width),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # No Photometric Augmentations (Brightness/Contrast) as per idea
                A.Normalize(
                    mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=1.0
                ),  # Just to ensure float32 range if needed, but we did manual norm.
                # Actually, our input is already 0-1 float32. A.Normalize usually subtracts mean/div std.
                # We don't want to mess with Age/Implant values.
                # So we skip A.Normalize or use identity.
                # We just need ToTensorV2 to convert HWC -> CHW.
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

    elif data == "val" or data == "test":
        return A.Compose(
            [A.Resize(height, width), ToTensorV2()],
            additional_targets={"image0": "image"},
        )


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main function to prepare dataloaders.
    """
    seed_everything(Config.SEED)

    # 1. Process Metadata
    df_train, df_val, df_test, age_stats = process_metadata(
        load_cached_data=load_cached_data
    )

    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)
        # Keep test full usually, or sample if strictly debugging pipeline
        # df_test = df_test.sample(n=100, random_state=Config.SEED).reset_index(drop=True)

    # 2. Create Datasets
    train_dataset = PairedBreastCancerDataset(
        df_train,
        Config.TRAIN_IMAGES_DIR,
        age_stats,
        transform=get_transforms("train"),
        is_test=False,
    )

    val_dataset = PairedBreastCancerDataset(
        df_val,
        Config.TRAIN_IMAGES_DIR,
        age_stats,
        transform=get_transforms("val"),
        is_test=False,
    )

    test_dataset = PairedBreastCancerDataset(
        df_test,
        Config.TEST_IMAGES_DIR,
        age_stats,
        transform=get_transforms("test"),
        is_test=True,
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
