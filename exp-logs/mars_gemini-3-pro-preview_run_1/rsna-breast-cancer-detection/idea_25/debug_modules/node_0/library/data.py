import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library import config

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_image(path):
    """
    Loads an image using OpenCV. Handles 8-bit and 16-bit depth.
    Normalizes to [0, 1].
    Raises FileNotFoundError if loading fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Load using UNCHANGED to preserve bit depth (often 16-bit for mammograms)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(
            f"Failed to load image (corrupt or unsupported format): {path}"
        )

    # Ensure single channel
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to 0-1 range based on dtype
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        # Fallback for other types, min-max normalization
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

    return img


def get_age_stats(df_train, load_cached_data=True):
    """
    Computes or loads mean and std of age from training data.
    """
    cache_path = os.path.join(config.CACHE_DIR, "age_stats.npy")

    if load_cached_data and os.path.exists(cache_path):
        stats = np.load(cache_path)
        return stats[0], stats[1]

    # Compute stats
    valid_ages = df_train["age"].dropna()
    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # Save to cache
    np.save(cache_path, np.array([mean_age, std_age]))

    return mean_age, std_age


def process_metadata(csv_path, split_name, load_cached_data=True):
    """
    Loads metadata and identifies contralateral pairs.
    Caches the processed dataframe to parquet.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load raw metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Logic to find contralateral pairs
    # We need to map (patient_id, view) -> {laterality: file_path}
    # This allows fast lookup of the "other" breast.

    # Create a lookup dictionary
    # Key: (patient_id, view), Value: {laterality: file_path}
    pair_lookup = {}

    for idx, row in df.iterrows():
        key = (row["patient_id"], row["view"])
        if key not in pair_lookup:
            pair_lookup[key] = {}

        # Construct full path relative to input dir
        # The metadata contains relative paths like "train_images/..."
        # We prepend INPUT_DIR here for easier loading later, or keep relative.
        # Let's keep relative as per metadata, but ensure we know where to look.
        pair_lookup[key][row["laterality"]] = row["file_path"]

    # Apply lookup to dataframe
    contra_paths = []

    for idx, row in df.iterrows():
        key = (row["patient_id"], row["view"])
        current_lat = row["laterality"]
        target_lat = "R" if current_lat == "L" else "L"

        # Check if counterpart exists
        if key in pair_lookup and target_lat in pair_lookup[key]:
            contra_paths.append(pair_lookup[key][target_lat])
        else:
            contra_paths.append(None)

    df["contra_file_path"] = contra_paths

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


# =============================================================================
# DATASET CLASS
# =============================================================================


class SiameseMammographyDataset(Dataset):
    def __init__(self, df, transforms=None, age_stats=(58.0, 10.0)):
        """
        Args:
            df (pd.DataFrame): Dataframe with 'file_path', 'contra_file_path', 'age', 'implant', 'cancer'.
            transforms (albumentations.Compose): Transforms to apply.
            age_stats (tuple): (mean, std) for age scaling.
        """
        self.df = df
        self.transforms = transforms
        self.age_mean, self.age_std = age_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_path = os.path.join(config.INPUT_DIR, row["file_path"])
        image = load_image(target_path)  # Returns normalized float32 (H, W)

        # 2. Load Contralateral Image
        contra_path_rel = row["contra_file_path"]
        if contra_path_rel is not None:
            contra_path = os.path.join(config.INPUT_DIR, contra_path_rel)
            try:
                image_contra = load_image(contra_path)
            except FileNotFoundError:
                # If listed in metadata but file missing, fail loud as per requirements
                raise FileNotFoundError(
                    f"Contralateral file listed but missing: {contra_path}"
                )
        else:
            # No physical contralateral image exists
            image_contra = np.zeros_like(image)

        # 3. Prepare Metadata Features
        # Age Scaling
        age = row["age"] if not pd.isna(row["age"]) else self.age_mean
        age_scaled = (age - self.age_mean) / (self.age_std + 1e-6)

        # Implant (Binary)
        implant = 1.0 if row["implant"] == 1 else 0.0

        # 4. Apply Synchronized Augmentations
        # We pass both images to albumentations to ensure same geometric transform
        if self.transforms:
            # Albumentations expects HWC or HW
            data = self.transforms(image=image, image_contra=image_contra)
            image = data["image"]
            image_contra = data["image_contra"]

        # 5. Construct 3-Channel Tensors
        # Input format: (3, H, W)
        # Channel 0: Image
        # Channel 1: Age Map
        # Channel 2: Implant Map

        def create_input_tensor(img_tensor, h, w):
            # img_tensor is (H, W) or (1, H, W) from ToTensorV2
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.squeeze(0)  # Ensure (H, W)

            # Create metadata maps
            age_map = torch.full((h, w), age_scaled, dtype=torch.float32)
            implant_map = torch.full((h, w), implant, dtype=torch.float32)

            # Stack
            return torch.stack([img_tensor, age_map, implant_map], dim=0)

        # Get dimensions after transform
        # If ToTensorV2 was used, these are tensors.
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            input_target = create_input_tensor(image, h, w)
            input_contra = create_input_tensor(image_contra, h, w)
        else:
            # Fallback if ToTensorV2 not in transforms (unlikely)
            h, w = image.shape[:2]
            image_t = torch.from_numpy(image).float()
            image_c_t = torch.from_numpy(image_contra).float()
            input_target = create_input_tensor(image_t, h, w)
            input_contra = create_input_tensor(image_c_t, h, w)

        # 6. Prepare Label and ID
        label = (
            torch.tensor(row["cancer"], dtype=torch.float32)
            if "cancer" in row
            else torch.tensor(-1.0)
        )
        prediction_id = (
            row["prediction_id"]
            if "prediction_id" in row
            else f"{row['patient_id']}_{row['laterality']}"
        )

        return {
            "image": input_target,  # (3, H, W)
            "image_contra": input_contra,  # (3, H, W)
            "label": label,
            "prediction_id": prediction_id,
        }


# =============================================================================
# TRANSFORMS & FACTORY
# =============================================================================


def get_transforms(phase):
    """
    Returns Albumentations transforms.
    Synchronized geometric transforms for Train.
    Resize/Pad for Val/Test.
    """
    height, width = config.IMAGE_SIZE

    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations (Synchronized)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                # Resize/Pad to fixed size
                A.Resize(height=height, width=width),
                # Convert to Tensor
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )

    else:
        return A.Compose(
            [
                # Deterministic Resize
                A.Resize(height=height, width=width),
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )


def get_dataloaders(load_cached_data=True, debug_subset_size=None):
    """
    Main entry point to get dataloaders.

    Args:
        load_cached_data (bool): Whether to use cached metadata/stats.
        debug_subset_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load and Process Metadata
    df_train = process_metadata(config.TRAIN_METADATA_PATH, "train", load_cached_data)
    df_val = process_metadata(config.VAL_METADATA_PATH, "val", load_cached_data)
    df_test = process_metadata(config.TEST_METADATA_PATH, "test", load_cached_data)

    # 2. Compute Age Statistics (from Train only)
    age_stats = get_age_stats(df_train, load_cached_data)

    # 3. Debugging Subset
    if debug_subset_size:
        df_train = df_train.iloc[:debug_subset_size]
        df_val = df_val.iloc[:debug_subset_size]
        df_test = df_test.iloc[:debug_subset_size]

    # 4. Create Datasets
    train_dataset = SiameseMammographyDataset(
        df_train, transforms=get_transforms("train"), age_stats=age_stats
    )

    val_dataset = SiameseMammographyDataset(
        df_val, transforms=get_transforms("val"), age_stats=age_stats
    )

    test_dataset = SiameseMammographyDataset(
        df_test, transforms=get_transforms("test"), age_stats=age_stats
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
