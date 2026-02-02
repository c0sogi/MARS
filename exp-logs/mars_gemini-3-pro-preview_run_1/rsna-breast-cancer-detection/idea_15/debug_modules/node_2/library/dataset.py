import os
import cv2
import numpy as np
import pandas as pd
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

logger = get_logger("dataset")


def load_image(path):
    """
    Loads an image from the given path using OpenCV.
    Handles DICOM/JPEG 2000 by extracting the codestream if direct loading fails.
    """
    # 0. Check for pre-processed .npy file (Cite debug_lesson_4)
    npy_path = path.replace(".dcm", ".npy")
    if os.path.exists(npy_path):
        try:
            return np.load(npy_path)
        except Exception:
            pass

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # 1. Try standard loading (works for PNG, JPG, and some JP2)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img

    # 2. Fallback: Try to extract JPEG/JP2 codestream from DICOM container (Cite debug_lesson_8)
    # This is required because pydicom is not available and cv2 cannot parse DICOM headers.
    try:
        with open(path, "rb") as f:
            content = f.read()

        # Common Magic Bytes (Cite debug_lesson_3)
        # JPEG: FF D8
        # JP2 Codestream: FF 4F FF 51
        # JP2 File: 00 00 00 0C 6A 50
        signatures = [
            b"\xff\xd8",  # JPEG
            b"\xff\x4f\xff\x51",  # JP2 Codestream
            b"\x00\x00\x00\x0c\x6a\x50",  # JP2 File
        ]

        start_idx = -1
        for sig in signatures:
            idx = content.find(sig)
            if idx != -1:
                if start_idx == -1 or idx < start_idx:
                    start_idx = idx

        if start_idx != -1:
            # Try cv2 first
            img_bytes = np.frombuffer(content[start_idx:], dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img

            # Fallback to PIL for JP2/JPEG (Cite debug_lesson_8)
            try:
                pil_img = Image.open(io.BytesIO(content[start_idx:]))
                return np.array(pil_img.convert("L"))
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"Manual decoding failed for {path}: {e}")

    # 3. Fail loudly if all methods fail
    raise FileNotFoundError(
        f"Failed to load image data (corrupt or unsupported format): {path}"
    )


def compute_age_stats(df, load_cached_data=True):
    """
    Computes or loads cached mean and std for the 'age' column.
    """
    cache_path = Config.CACHE_AGE_STATS

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading age stats from {cache_path}")
        stats = np.load(cache_path, allow_pickle=True).item()
        return stats["mean"], stats["std"]

    logger.info("Computing age stats from training data...")
    # Fill missing age with median before stats or just dropna for stats?
    # Usually better to compute stats on valid data.
    valid_ages = df["age"].dropna()
    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, {"mean": mean_age, "std": std_age})

    logger.info(f"Computed Age Mean: {mean_age:.4f}, Std: {std_age:.4f}")
    return mean_age, std_age


def process_metadata(df, dataset_type, load_cached_data=True):
    """
    Processes metadata to identify contralateral pairs.
    Caches the processed dataframe to parquet.

    Args:
        df (pd.DataFrame): Raw metadata.
        dataset_type (str): 'train', 'val', or 'test' for cache naming.
        load_cached_data (bool): Whether to use cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'contra_file_path'.
    """
    if dataset_type == "train":
        cache_path = Config.CACHE_TRAIN_PATH
    elif dataset_type == "val":
        cache_path = Config.CACHE_VAL_PATH
    else:
        cache_path = Config.CACHE_TEST_PATH

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading processed {dataset_type} metadata from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Processing {dataset_type} metadata for contralateral pairing...")

    # Ensure necessary columns exist
    required_cols = ["patient_id", "view", "laterality", "file_path"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in metadata")

    # Create a lookup for file paths: (patient_id, view, laterality) -> file_path
    # We take the first image found if duplicates exist (determinism)
    lookup = (
        df.groupby(["patient_id", "view", "laterality"])["file_path"].first().to_dict()
    )

    def get_contra_path(row):
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]

        # Determine opposite laterality
        contra_lat = "R" if lat == "L" else "L"

        # Look up
        key = (pid, view, contra_lat)
        return lookup.get(key, None)

    df["contra_file_path"] = df.apply(get_contra_path, axis=1)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    found_count = df["contra_file_path"].notna().sum()
    logger.info(f"Processed {len(df)} rows. Found {found_count} contralateral pairs.")

    return df


class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None, age_mean=0.0, age_std=1.0, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transforms (albumentations.Compose): Augmentation pipeline.
            age_mean (float): Mean age for normalization.
            age_std (float): Std age for normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.age_mean = age_mean
        self.age_std = age_std
        self.mode = mode

        # Pre-calculate normalized age and implant for efficiency
        # Handle missing ages by filling with mean (0 after normalization)
        ages = self.df["age"].fillna(self.age_mean).values
        self.norm_ages = (ages - self.age_mean) / (self.age_std + 1e-6)

        # Handle implant (binary 0/1)
        if "implant" in self.df.columns:
            self.implants = self.df["implant"].fillna(0).astype(np.float32).values
        else:
            self.implants = np.zeros(len(self.df), dtype=np.float32)

        # Labels
        if "cancer" in self.df.columns:
            self.labels = self.df["cancer"].values.astype(np.float32)
        else:
            self.labels = None

        # Paths
        self.file_paths = self.df["file_path"].values
        self.contra_paths = self.df["contra_file_path"].values

        # Prediction IDs for test
        if "prediction_id" in self.df.columns:
            self.prediction_ids = self.df["prediction_id"].values
        else:
            self.prediction_ids = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Target Image
        img_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])
        image = load_image(img_path)  # Returns 2D grayscale

        # 2. Load Contralateral Image
        contra_path_rel = self.contra_paths[idx]
        if contra_path_rel is not None:
            contra_full_path = os.path.join(Config.INPUT_DIR, contra_path_rel)
            try:
                image_contra = load_image(contra_full_path)
            except FileNotFoundError:
                # If specifically contralateral is corrupt/missing despite path existing in csv,
                # treat as missing contralateral (zero tensor)
                image_contra = np.zeros_like(image)
        else:
            # No contralateral exists -> Zero Tensor
            image_contra = np.zeros_like(image)

        # Ensure dimensions match (just in case of weird data)
        if image.shape != image_contra.shape:
            image_contra = cv2.resize(image_contra, (image.shape[1], image.shape[0]))

        # 3. Apply Transforms (Synchronized)
        # We pass both images to albumentations to ensure same geometric transform
        if self.transforms:
            augmented = self.transforms(image=image, image_contra=image_contra)
            image = augmented["image"]
            image_contra = augmented["image_contra"]

        # Note: ToTensorV2 converts to [C, H, W] and scales if specified.
        # Here we assume images are 0-255 uint8 coming in.
        # If ToTensorV2 is used without normalization in Compose, it just converts to Tensor.
        # We need to normalize to [0, 1] float manually if not done by transforms.

        if isinstance(image, torch.Tensor):
            image = image.float() / 255.0
            image_contra = image_contra.float() / 255.0
        else:
            # Fallback if transforms didn't convert to tensor
            image = torch.from_numpy(image).float() / 255.0
            image_contra = torch.from_numpy(image_contra).float() / 255.0

        # Ensure channel dim exists (1, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image_contra.dim() == 2:
            image_contra = image_contra.unsqueeze(0)

        # 4. Construct Metadata Maps (Age & Implant)
        # Spatially broadcast scalars to (1, H, W)
        H, W = image.shape[1], image.shape[2]

        age_val = self.norm_ages[idx]
        implant_val = self.implants[idx]

        age_map = torch.full((1, H, W), age_val, dtype=torch.float32)
        implant_map = torch.full((1, H, W), implant_val, dtype=torch.float32)

        # 5. Stack Channels
        # Input: [Image, Age, Implant] -> (3, H, W)
        target_tensor = torch.cat([image, age_map, implant_map], dim=0)

        # Contra Input: [ContraImage, Age, Implant] -> (3, H, W)
        # Note: Age and Implant are patient-level, so they are same for contra.
        # Even if contra image is zero, we pass age/implant to allow demographic cancellation.
        contra_tensor = torch.cat([image_contra, age_map, implant_map], dim=0)

        # 6. Return
        sample = {
            "image": target_tensor,
            "image_contra": contra_tensor,
        }

        if self.labels is not None:
            # Fix: Unsqueeze to [1] so batching creates [B, 1] (Cite debug_lesson_12)
            sample["label"] = torch.tensor(
                self.labels[idx], dtype=torch.float32
            ).unsqueeze(0)

        if self.prediction_ids is not None:
            sample["prediction_id"] = self.prediction_ids[idx]

        return sample


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Synchronized geometric transforms for paired images.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),  # D4 symmetry
                # Conservative geometric augs
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    Handles metadata loading, caching, and dataset instantiation.
    """
    # 1. Load Metadata
    train_df_raw = pd.read_csv(Config.TRAIN_METADATA)
    val_df_raw = pd.read_csv(Config.VAL_METADATA)
    test_df_raw = pd.read_csv(Config.TEST_METADATA)

    # 2. Compute/Load Age Stats (from Train only)
    age_mean, age_std = compute_age_stats(
        train_df_raw, load_cached_data=load_cached_data
    )

    # 3. Process Metadata (Pairing)
    train_df = process_metadata(
        train_df_raw, "train", load_cached_data=load_cached_data
    )
    val_df = process_metadata(val_df_raw, "val", load_cached_data=load_cached_data)
    test_df = process_metadata(test_df_raw, "test", load_cached_data=load_cached_data)

    # 4. Create Datasets
    train_dataset = BreastCancerDataset(
        train_df,
        transforms=get_transforms("train"),
        age_mean=age_mean,
        age_std=age_std,
        mode="train",
    )

    val_dataset = BreastCancerDataset(
        val_df,
        transforms=get_transforms("val"),
        age_mean=age_mean,
        age_std=age_std,
        mode="val",
    )

    test_dataset = BreastCancerDataset(
        test_df,
        transforms=get_transforms("test"),
        age_mean=age_mean,
        age_std=age_std,
        mode="test",
    )

    # 5. Create DataLoaders
    # Use num_workers from Config
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,  # Cite debug_lesson_9
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,  # Cite debug_lesson_9
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,  # Cite debug_lesson_9
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
