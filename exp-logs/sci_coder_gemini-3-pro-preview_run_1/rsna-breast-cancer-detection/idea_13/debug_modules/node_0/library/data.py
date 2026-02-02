import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(phase: str, img_size: tuple):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train' or 'test'/'val'.
        img_size (tuple): (height, width).
    """
    if phase == "train":
        return A.Compose(
            [
                # Geometric augmentations only, as per strategy
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                A.Resize(height=img_size[0], width=img_size[1]),
                # Normalize image to 0-1 (assuming 8-bit input)
                A.Normalize(
                    mean=(0,), std=(1,), max_pixel_value=255.0, always_apply=True
                ),
                ToTensorV2(),
            ],
            additional_targets={"image_c": "image"},
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(
                    mean=(0,), std=(1,), max_pixel_value=255.0, always_apply=True
                ),
                ToTensorV2(),
            ],
            additional_targets={"image_c": "image"},
        )


def compute_and_cache_age_stats(df_train, cache_dir, load_cached=True):
    """
    Computes mean and std of age from training data and caches it.

    Args:
        df_train (pd.DataFrame): Training metadata.
        cache_dir (str): Directory to save the stats.
        load_cached (bool): Whether to try loading from cache.

    Returns:
        tuple: (mean_age, std_age)
    """
    stats_path = os.path.join(cache_dir, "age_stats.npy")

    if load_cached and os.path.exists(stats_path):
        try:
            stats = np.load(stats_path)
            return stats[0], stats[1]
        except Exception as e:
            print(f"Failed to load cached age stats: {e}. Recomputing...")

    # Compute stats
    # Handle missing values by imputing with temporary mean
    ages = df_train["age"].copy()
    temp_mean = ages.mean()
    ages = ages.fillna(temp_mean)

    mean_age = ages.mean()
    std_age = ages.std()

    # Save
    os.makedirs(cache_dir, exist_ok=True)
    np.save(stats_path, np.array([mean_age, std_age]))

    return mean_age, std_age


class BreastCancerDataset(Dataset):
    def __init__(
        self, df, input_dir, transform=None, age_stats=(58.0, 10.0), is_test=False
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            input_dir (str): Root directory for images.
            transform (A.Compose): Albumentations transforms.
            age_stats (tuple): (mean_age, std_age) for normalization.
            is_test (bool): Whether this is the test set (returns prediction_id).
        """
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.transform = transform
        self.mean_age, self.std_age = age_stats
        self.is_test = is_test

        # Build lookup for contralateral images
        # Key: (patient_id, view, laterality) -> Value: file_path
        self.lookup = {}
        for idx, row in self.df.iterrows():
            key = (row["patient_id"], row["view"], row["laterality"])
            self.lookup[key] = row["file_path"]

        # Also need a global lookup for the whole dataset if the df passed is just a split (e.g. train vs val)
        # However, typically train/val splits are by patient, so the contralateral should be in the same split.
        # If not, we might miss it. But strictly following the split is safer to avoid leakage.
        # We assume the contralateral image is available in the provided df or we accept it as missing.
        # Optimization: To ensure we find contralateral even if it's not in the current batch/split (though it should be for patient split),
        # we ideally need access to the full index. But for this implementation, we rely on the passed df.

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        full_path = os.path.join(self.input_dir, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Load as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fail loudly as requested
            raise ValueError(
                f"Failed to load image (corrupt or unsupported format): {full_path}"
            )

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_path = row["file_path"]
        img_target = self._load_image(target_path)

        # 2. Find and Load Contralateral Image
        # Logic: Same patient, same view, opposite laterality
        opp_laterality = "R" if row["laterality"] == "L" else "L"
        contra_key = (row["patient_id"], row["view"], opp_laterality)

        if contra_key in self.lookup:
            contra_path = self.lookup[contra_key]
            try:
                img_contra = self._load_image(contra_path)
            except (FileNotFoundError, ValueError):
                # If contralateral file is listed but corrupt/missing, treat as missing
                img_contra = np.zeros_like(img_target)
        else:
            # Physically missing
            img_contra = np.zeros_like(img_target)

        # Ensure dimensions match (in case of different image sizes before resize)
        # We let albumentations handle resizing, but if we use zeros, we should match target shape
        if img_contra.shape != img_target.shape:
            img_contra = cv2.resize(
                img_contra,
                (img_target.shape[1], img_target.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # 3. Apply Synchronized Augmentation
        if self.transform:
            # Pass both images to the transform to ensure same geometric ops
            augmented = self.transform(image=img_target, image_c=img_contra)
            img_target_tensor = augmented["image"]  # (1, H, W) float tensor
            img_contra_tensor = augmented["image_c"]  # (1, H, W) float tensor
        else:
            # Fallback to simple tensor conversion
            img_target_tensor = (
                torch.from_numpy(img_target).float().unsqueeze(0) / 255.0
            )
            img_contra_tensor = (
                torch.from_numpy(img_contra).float().unsqueeze(0) / 255.0
            )

        # 4. Create Metadata Channels (Age & Implant)
        # Age: Standard Scaled
        age_val = row["age"]
        if pd.isna(age_val):
            age_val = self.mean_age  # Impute with global mean

        norm_age = (age_val - self.mean_age) / (self.std_age + 1e-7)

        # Implant: Binary (0 or 1)
        implant_val = 1.0 if row["implant"] == 1 else 0.0

        # Create spatially broadcasted maps
        # Tensor shape is (C, H, W). We need to append to the existing (1, H, W) image tensor
        _, h, w = img_target_tensor.shape

        age_map = torch.full((1, h, w), norm_age, dtype=torch.float32)
        implant_map = torch.full((1, h, w), implant_val, dtype=torch.float32)

        # 5. Stack Channels -> (3, H, W)
        # Structure: [Image, Age, Implant]
        target_input = torch.cat([img_target_tensor, age_map, implant_map], dim=0)
        contra_input = torch.cat([img_contra_tensor, age_map, implant_map], dim=0)

        # 6. Return Data
        if self.is_test:
            return target_input, contra_input, row["prediction_id"]
        else:
            label = torch.tensor(row["cancer"], dtype=torch.float32)
            return target_input, contra_input, label


def prepare_data(load_cached_data=True):
    """
    Loads metadata and prepares age statistics.

    Args:
        load_cached_data (bool): Whether to use cached stats.

    Returns:
        tuple: (df_train, df_val, df_test, age_stats)
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Compute/Load Age Stats
    # We use only training data for stats to avoid leakage
    age_stats = compute_and_cache_age_stats(
        df_train, Config.WORKING_DIR, load_cached=load_cached_data
    )

    return df_train, df_val, df_test, age_stats


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create dataloaders for train, val, and test.

    Args:
        load_cached_data (bool): Whether to use cached metadata/stats.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    df_train, df_val, df_test, age_stats = prepare_data(load_cached_data)

    # Transforms
    train_transform = get_transforms("train", Config.IMG_SIZE)
    eval_transform = get_transforms("val", Config.IMG_SIZE)

    # Datasets
    train_dataset = BreastCancerDataset(
        df_train,
        Config.INPUT_DIR,
        transform=train_transform,
        age_stats=age_stats,
        is_test=False,
    )

    val_dataset = BreastCancerDataset(
        df_val,
        Config.INPUT_DIR,
        transform=eval_transform,
        age_stats=age_stats,
        is_test=False,
    )

    test_dataset = BreastCancerDataset(
        df_test,
        Config.INPUT_DIR,
        transform=eval_transform,
        age_stats=age_stats,
        is_test=True,
    )

    # DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
