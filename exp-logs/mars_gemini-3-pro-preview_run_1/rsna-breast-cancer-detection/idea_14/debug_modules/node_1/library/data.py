import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_age_stats(df_train, load_cached_data=True):
    """
    Computes or loads age statistics (mean, std) for normalization.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "age_stats.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path)
            return stats[0], stats[1]
        except Exception as e:
            print(f"Failed to load cached age stats: {e}. Recomputing.")

    # Compute stats ignoring NaNs
    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()

    # Save
    try:
        np.save(cache_path, np.array([age_mean, age_std]))
    except Exception as e:
        print(f"Warning: Could not save age stats to cache: {e}")

    return age_mean, age_std


def process_metadata(df, split_name, load_cached_data=True):
    """
    Processes metadata to add contralateral file paths.
    Uses caching to speed up subsequent runs.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df_cached = pd.read_parquet(cache_path)
            # Basic validation to ensure cache matches input size roughly
            if len(df_cached) == len(df):
                return df_cached
        except Exception as e:
            print(
                f"Failed to load cached metadata for {split_name}: {e}. Reprocessing."
            )

    # Create lookup for pairing: (patient_id, view, laterality) -> file_path
    # We drop duplicates to ensure unique mapping; usually (patient, view, lat) is unique per image
    # or we just take the first one available.
    lookup_df = df[["patient_id", "view", "laterality", "file_path"]].drop_duplicates(
        subset=["patient_id", "view", "laterality"]
    )

    # Create a dictionary for O(1) lookup
    # Key: (patient_id, view, laterality)
    lookup_dict = lookup_df.set_index(["patient_id", "view", "laterality"])[
        "file_path"
    ].to_dict()

    def find_contra(row):
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]
        # Determine opposite laterality
        opp_lat = "R" if lat == "L" else "L"

        # Look up
        return lookup_dict.get((pid, view, opp_lat), None)

    # Apply pairing logic
    df["contra_file_path"] = df.apply(find_contra, axis=1)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save processed metadata to {cache_path}: {e}")

    return df


def filter_existing_files(df, img_dir):
    """
    Filters the dataframe to include only rows where the image file exists.
    """
    exists = [os.path.exists(os.path.join(img_dir, fp)) for fp in df["file_path"]]

    initial_len = len(df)
    filtered_df = df[exists].reset_index(drop=True)
    final_len = len(filtered_df)

    print(
        f"Filtered dataset: {initial_len} -> {final_len} samples (Removed {initial_len - final_len} missing files)"
    )
    return filtered_df


def get_transforms(mode="train", img_size=Config.IMG_SIZE):
    """
    Returns Albumentations transforms.
    Synchronized transforms for 'image' and 'image_contra' are handled by passing
    additional_targets to the Compose object or handling it in the Dataset.
    Here we define the pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                # Geometric augmentations (Synchronized)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                # No photometric augmentations (brightness/contrast) as density is key
                A.Normalize(
                    mean=(0.0,), std=(1.0,), max_pixel_value=1.0
                ),  # Just scaling 0-1 if input is 0-1
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=1.0),
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )


class BreastCancerDataset(Dataset):
    def __init__(self, df, img_dir, age_stats, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.age_mean, self.age_std = age_stats
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        if rel_path is None or pd.isna(rel_path):
            return None

        full_path = os.path.join(self.img_dir, rel_path)
        if not os.path.exists(full_path):
            return None

        # Use imdecode to handle potential extension mismatches or issues
        try:
            # Read as binary
            with open(full_path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if img is None:
                return None

            # Handle grayscale vs RGB (we expect grayscale)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Normalize to 0-1 float
            img = img.astype(np.float32)
            if img.max() > 0:
                img /= 255.0
            else:
                img /= 1.0  # Avoid div by zero for empty images

            return img
        except Exception:
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_path = row["file_path"]
        img_target = self._load_image(target_path)

        if img_target is None:
            # Fail Loudly as requested
            raise FileNotFoundError(f"Could not load target image: {target_path}")

        # 2. Load Contralateral Image
        contra_path = row["contra_file_path"]
        img_contra = self._load_image(contra_path)

        # If contra missing, use zeros matching target shape
        if img_contra is None:
            img_contra = np.zeros_like(img_target)

        # Ensure shapes match exactly before transform (resize will handle final size)
        # But if raw sizes differ significantly, resize contra to target first?
        # Albumentations Resize will handle it.

        # 3. Apply Transforms (Synchronized)
        if self.transforms:
            augmented = self.transforms(image=img_target, image_contra=img_contra)
            img_target = augmented["image"]
            img_contra = augmented["image_contra"]

        # 4. Construct Metadata Channels
        # Age
        age = row["age"]
        if pd.isna(age):
            age = self.age_mean
        age_norm = (age - self.age_mean) / (self.age_std + 1e-7)

        # Implant (0 or 1)
        implant = row["implant"] if "implant" in row else 0
        if pd.isna(implant):
            implant = 0
        implant = float(implant)

        # Create maps (C, H, W) -> (1, H, W)
        # img_target is tensor (1, H, W) or (H, W) depending on ToTensorV2
        # ToTensorV2 produces (C, H, W) if input is (H, W, C) or (H, W).
        # For grayscale (H, W), ToTensorV2 adds no channel dim by default?
        # Actually ToTensorV2 converts HWC->CHW. If HW, it returns HW tensor?
        # Let's check shape.
        if img_target.ndim == 2:
            img_target = img_target.unsqueeze(0)
        if img_contra.ndim == 2:
            img_contra = img_contra.unsqueeze(0)

        _, h, w = img_target.shape

        age_map = torch.full((1, h, w), age_norm, dtype=torch.float32)
        implant_map = torch.full((1, h, w), implant, dtype=torch.float32)

        # 5. Stack Channels -> (3, H, W)
        # Target Input
        target_tensor = torch.cat([img_target, age_map, implant_map], dim=0)

        # Contra Input (Age/Implant are same for patient)
        contra_tensor = torch.cat([img_contra, age_map, implant_map], dim=0)

        # 6. Label
        label = row["cancer"] if "cancer" in row else -1
        label = torch.tensor(label, dtype=torch.float32)

        return target_tensor, contra_tensor, label


def get_dataloaders(
    load_cached_data=True,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Main function to prepare DataLoaders.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # 2. Compute/Load Stats
    age_stats = get_age_stats(df_train, load_cached_data=load_cached_data)

    # 3. Process Metadata (Pairing)
    df_train = process_metadata(df_train, "train", load_cached_data=load_cached_data)
    df_val = process_metadata(df_val, "val", load_cached_data=load_cached_data)
    df_test = process_metadata(df_test, "test", load_cached_data=load_cached_data)

    # --- FILTER MISSING FILES ---
    print("Filtering datasets for existing files...")
    df_train = filter_existing_files(df_train, Config.INPUT_DIR)
    df_val = filter_existing_files(df_val, Config.INPUT_DIR)
    df_test = filter_existing_files(df_test, Config.INPUT_DIR)

    # 4. Debug Subsampling
    if debug:
        n_train = min(len(df_train), debug_sample_size)
        n_val = min(len(df_val), debug_sample_size)
        n_test = min(len(df_test), debug_sample_size)

        if n_train > 0:
            df_train = df_train.sample(n=n_train, random_state=Config.SEED).reset_index(
                drop=True
            )
        if n_val > 0:
            df_val = df_val.sample(n=n_val, random_state=Config.SEED).reset_index(
                drop=True
            )
        if n_test > 0:
            df_test = df_test.sample(n=n_test, random_state=Config.SEED).reset_index(
                drop=True
            )

        print(
            f"DEBUG MODE: Subsampled datasets to {len(df_train)} train, {len(df_val)} val."
        )

    # 5. Create Datasets
    train_dataset = BreastCancerDataset(
        df_train, Config.INPUT_DIR, age_stats, transforms=get_transforms("train")
    )

    val_dataset = BreastCancerDataset(
        df_val, Config.INPUT_DIR, age_stats, transforms=get_transforms("val")
    )

    test_dataset = BreastCancerDataset(
        df_test, Config.INPUT_DIR, age_stats, transforms=get_transforms("test")
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True if len(train_dataset) >= Config.BATCH_SIZE else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
