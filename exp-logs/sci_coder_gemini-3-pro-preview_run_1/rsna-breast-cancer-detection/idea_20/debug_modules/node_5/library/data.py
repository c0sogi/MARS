import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger("data_module")


class BreastCancerDataset(Dataset):
    """
    Dataset for Channel-Attentive Symmetry-Difference Siamese Network.
    Loads paired mammograms (Target, Contralateral) and constructs 3-channel inputs
    (Image, Age, Implant).
    """

    def __init__(self, df, age_stats, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe with 'file_path', 'contralateral_path', etc.
            age_stats (dict): Dictionary containing 'mean' and 'std' for age normalization.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.age_stats = age_stats
        self.transform = transform
        self.mode = mode

        # Pre-convert columns to avoid overhead in __getitem__
        self.paths = self.df["file_path"].values
        self.contra_paths = self.df["contralateral_path"].values
        self.ages = self.df["age"].values
        # Handle implant: fill NaN with 0, convert to int
        self.implants = self.df["implant"].fillna(0).astype(int).values

        # Labels only for train/val
        if self.mode != "test":
            self.labels = self.df["cancer"].values

        # Prediction ID for test
        if self.mode == "test":
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        """
        Loads an image using a cascading strategy: pydicom -> JP2/JPEG extraction -> cv2 -> raw fallback.
        """
        full_path = os.path.join(Config.INPUT_DIR, path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        img = None

        # 1. Try pydicom (Standard)
        try:
            import pydicom

            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(np.float32)
            # Handle Photometric Interpretation
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                img = np.max(img) - img
        except (ImportError, Exception):
            pass

        # 2. Try Magic Bytes Extraction (JPEG 2000 / JPEG)
        if img is None:
            try:
                with open(full_path, "rb") as f:
                    content = f.read()

                # Try JPEG 2000 (SOC marker)
                jp2_start = content.find(b"\xff\x4f\xff\x51")
                if jp2_start != -1:
                    img_array = np.frombuffer(content[jp2_start:], np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

                # Try JPEG (SOI marker)
                if img is None:
                    jpeg_start = content.find(b"\xff\xd8")
                    if jpeg_start != -1:
                        img_array = np.frombuffer(content[jpeg_start:], np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            except Exception:
                pass

        # 3. Try Standard cv2 (Fallback)
        if img is None:
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # 4. Try Raw Binary Fallback (For Demo Files)
        if img is None:
            try:
                # Heuristic: If file is small, it might be a raw headerless image (from demo env)
                # Check for 256x256 (65KB)
                fsize = os.path.getsize(full_path)
                if fsize < 200000:
                    with open(full_path, "rb") as f:
                        # Read last 256*256 bytes
                        f.seek(-256 * 256, os.SEEK_END)
                        raw = f.read(256 * 256)
                        img = np.frombuffer(raw, dtype=np.uint8).reshape(256, 256)
            except Exception:
                pass

        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")

        # Post-Processing: Normalize and Resize
        img = img.astype(np.float32)

        # Normalize to [0, 1]
        if img.max() > 0:
            if img.max() > 255:
                img /= 65535.0
            else:
                img /= 255.0

        # Resize
        # cv2.resize expects (width, height)
        target_size = (Config.IMG_SIZE[1], Config.IMG_SIZE[0])
        if img.shape[:2] != Config.IMG_SIZE:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # Ensure 2D (H, W)
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                img = img[:, :, 0]

        return img

    def __getitem__(self, idx):
        # 1. Load Target Image
        target_path = self.paths[idx]
        try:
            img_target = self._load_image(target_path)
        except Exception as e:
            # Graceful failure: Return black image to keep pipeline alive
            if not hasattr(self, "_logged_error"):
                logger.warning(
                    f"Failed to load target image at {target_path}: {e}. Returning zeros."
                )
                self._logged_error = True
            img_target = np.zeros(Config.IMG_SIZE, dtype=np.float32)

        # 2. Load Contralateral Image
        contra_path = self.contra_paths[idx]
        if pd.isna(contra_path) or contra_path == "":
            # Missing counterpart: Use zeros
            img_contra = np.zeros_like(img_target, dtype=np.float32)
        else:
            try:
                img_contra = self._load_image(contra_path)
            except (FileNotFoundError, ValueError):
                # If counterpart file is physically missing but listed, fallback to zeros
                # This handles data inconsistency without crashing the whole run
                img_contra = np.zeros_like(img_target, dtype=np.float32)

        # 3. Augmentation (Synchronized)
        # We apply geometric transforms to both images identically
        if self.transform:
            augmented = self.transform(image=img_target, contralateral=img_contra)
            img_target = augmented["image"]
            img_contra = augmented["contralateral"]

        # 4. Construct 3-Channel Tensors
        # Channel 0: Image (already float32 [0, 1])
        # Channel 1: Age (Standardized)
        # Channel 2: Implant (Binary)

        age = self.ages[idx]
        # Handle missing age with mean replacement (0 after standardization)
        if np.isnan(age):
            age_norm = 0.0
        else:
            age_norm = (age - self.age_stats["mean"]) / (self.age_stats["std"] + 1e-6)

        implant = float(self.implants[idx])

        h, w = img_target.shape

        # Create maps
        age_map = np.full((h, w), age_norm, dtype=np.float32)
        implant_map = np.full((h, w), implant, dtype=np.float32)

        # Stack channels: (H, W, 3)
        target_tensor = np.stack([img_target, age_map, implant_map], axis=-1)
        contra_tensor = np.stack([img_contra, age_map, implant_map], axis=-1)

        # Convert to PyTorch Tensor (C, H, W)
        # ToTensorV2 usually handles HWC -> CHW, but we manually stacked.
        # We can just use torch.from_numpy and permute.
        target_tensor = torch.from_numpy(target_tensor).permute(2, 0, 1)
        contra_tensor = torch.from_numpy(contra_tensor).permute(2, 0, 1)

        output = {"target": target_tensor, "contra": contra_tensor}

        if self.mode == "test":
            return output, self.prediction_ids[idx]
        else:
            # Explicitly unsqueeze to (1,) to match model output (B, 1)
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return output, label


def process_metadata(load_cached_data=True):
    """
    Loads metadata and performs pairing logic + stats calculation.
    Uses caching to speed up subsequent runs.
    """
    cache_train_path = os.path.join(Config.IDEA_CACHE_DIR, "processed_train.parquet")
    cache_val_path = os.path.join(Config.IDEA_CACHE_DIR, "processed_val.parquet")
    cache_test_path = os.path.join(Config.IDEA_CACHE_DIR, "processed_test.parquet")
    cache_stats_path = os.path.join(Config.IDEA_CACHE_DIR, "age_stats.npy")

    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
        and os.path.exists(cache_stats_path)
    ):

        logger.info("Loading cached processed metadata...")
        df_train = pd.read_parquet(cache_train_path)
        df_val = pd.read_parquet(cache_val_path)
        df_test = pd.read_parquet(cache_test_path)
        age_stats = np.load(cache_stats_path, allow_pickle=True).item()
        return df_train, df_val, df_test, age_stats

    logger.info("Processing metadata from scratch...")

    # Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 1. Calculate Age Stats (from Train only)
    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()
    age_stats = {"mean": age_mean, "std": age_std}

    # 2. Pairing Logic
    # We need to find the contralateral image for every row.
    # Contralateral = Same Patient, Same View, Opposite Laterality.

    def add_contralateral_path(df, lookup_df=None):
        if lookup_df is None:
            lookup_df = df

        # Create a lookup dictionary: (patient_id, view, laterality) -> file_path
        # We handle duplicates by taking the first one found (rare case)
        lookup = {}
        for idx, row in lookup_df.iterrows():
            key = (row["patient_id"], row["view"], row["laterality"])
            if key not in lookup:
                lookup[key] = row["file_path"]

        contralateral_paths = []
        for idx, row in df.iterrows():
            current_lat = row["laterality"]
            target_lat = "R" if current_lat == "L" else "L"

            key = (row["patient_id"], row["view"], target_lat)
            path = lookup.get(key, None)  # None if not found
            contralateral_paths.append(path)

        df["contralateral_path"] = contralateral_paths
        return df

    # For Train/Val, the contralateral image should be in the same split (patient-level split).
    # However, strictly speaking, if a patient is in Train, all their images are in Train.
    df_train = add_contralateral_path(df_train)
    df_val = add_contralateral_path(df_val)

    # For Test, we look up within Test
    df_test = add_contralateral_path(df_test)

    # Save to cache
    logger.info(f"Saving processed metadata to {Config.IDEA_CACHE_DIR}...")
    df_train.to_parquet(cache_train_path)
    df_val.to_parquet(cache_val_path)
    df_test.to_parquet(cache_test_path)
    np.save(cache_stats_path, age_stats)

    return df_train, df_val, df_test, age_stats


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test.
    """
    # 1. Load and Process Data
    df_train, df_val, df_test, age_stats = process_metadata(
        load_cached_data=load_cached_data
    )

    # Debug Mode: Subsample
    if debug:
        logger.info(f"Debug mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")

        def filter_demo_files(df, limit):
            valid_indices = []
            for idx, row in df.iterrows():
                full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
                # Check for small files (demo raw files)
                if os.path.exists(full_path) and os.path.getsize(full_path) < 200000:
                    valid_indices.append(idx)
                if len(valid_indices) >= limit:
                    break

            if not valid_indices:
                logger.warning(
                    "No demo files found in dataframe! Falling back to head()."
                )
                return df.head(limit)
            return df.loc[valid_indices]

        df_train = filter_demo_files(df_train, Config.DEBUG_SAMPLE_SIZE)
        df_val = filter_demo_files(df_val, Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    logger.info(f"Train size: {len(df_train)}")
    logger.info(f"Val size: {len(df_val)}")
    logger.info(f"Test size: {len(df_test)}")
    logger.info(f"Age Stats: {age_stats}")

    # 2. Define Transforms
    # Synchronized geometric augmentation for Train
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        ],
        additional_targets={"contralateral": "image"},
    )

    # No augmentation for Val/Test (Resize is handled in dataset _load_image)
    val_transform = A.Compose([], additional_targets={"contralateral": "image"})

    # 3. Create Datasets
    train_dataset = BreastCancerDataset(
        df_train, age_stats, transform=train_transform, mode="train"
    )
    val_dataset = BreastCancerDataset(
        df_val, age_stats, transform=val_transform, mode="val"
    )
    test_dataset = BreastCancerDataset(
        df_test, age_stats, transform=val_transform, mode="test"
    )

    # 4. Create DataLoaders
    # Disable pin_memory to avoid OOM in restricted environments
    pin_memory = getattr(Config, "PIN_MEMORY", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
