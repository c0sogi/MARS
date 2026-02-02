import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import configuration and utilities from the provided library
from library.config import Config
from library.utils import get_logger

# Attempt to import pydicom
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logger = get_logger("data")


def load_dicom_slice(path, img_size=None):
    """
    Loads a single DICOM slice, applies bone windowing, and resizes.
    Returns a numpy array (H, W) normalized to 0-255 (uint8).
    """
    if not os.path.exists(path):
        # Fallback for missing files (should not happen in valid dataset)
        return (
            np.zeros((img_size, img_size), dtype=np.uint8)
            if img_size
            else np.zeros((256, 256), dtype=np.uint8)
        )

    try:
        if HAS_PYDICOM:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            # Apply Rescale Slope/Intercept if present
            slope = getattr(dcm, "RescaleSlope", 1.0)
            intercept = getattr(dcm, "RescaleIntercept", 0.0)
            img = img * slope + intercept
        else:
            # Fallback if pydicom is missing (e.g. if files are just images)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {path}")
            img = img.astype(np.float32)

        # Apply Bone Windowing
        center = Config.WINDOW_CENTER
        width = Config.WINDOW_WIDTH
        lower = center - width // 2
        upper = center + width // 2

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
        img = (img * 255.0).astype(np.uint8)

        # Resize
        if img_size and (img.shape[0] != img_size or img.shape[1] != img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        return img

    except Exception as e:
        # logger.warning(f"Error loading {path}: {e}")
        return (
            np.zeros((img_size, img_size), dtype=np.uint8)
            if img_size
            else np.zeros((256, 256), dtype=np.uint8)
        )


class RSNADataset(Dataset):
    def __init__(
        self, metadata_df, subset="train", load_cached_data=True, transform=None
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            subset (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load/save .npy files from Config.CACHE_DIR.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = metadata_df
        self.subset = subset
        self.load_cached_data = load_cached_data
        self.transform = transform

        # Ensure cache directory exists
        if self.load_cached_data:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # 1. Load Volume (from cache or process from scratch)
        volume = self._get_volume(row)  # Shape: (64, H, W, 3) uint8

        # 2. Apply Volumetric-Consistent Augmentation
        if self.transform:
            volume = self._apply_consistent_augmentations(volume)
        else:
            # Just normalize and convert to tensor if no transforms provided
            volume = volume.astype(np.float32) / 255.0
            volume = torch.tensor(volume).permute(0, 3, 1, 2)  # (D, C, H, W)

        # 3. Get Labels
        labels = self._get_labels(row)

        return volume, labels

    def _get_volume(self, row):
        study_id = row["StudyInstanceUID"]
        cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing

        # Process from scratch
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Get all slice files
        slice_files = glob.glob(os.path.join(image_dir, "*"))
        # Sort by slice number (assuming filename is number.dcm or similar)
        # Extract number from filename
        try:
            slice_files = sorted(
                slice_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
        except ValueError:
            slice_files = sorted(slice_files)  # Fallback lexicographical sort

        num_files = len(slice_files)
        if num_files == 0:
            # Return empty volume
            return np.zeros(
                (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),
                dtype=np.uint8,
            )

        # Uniform Temporal Subsampling
        indices = np.linspace(0, num_files - 1, Config.NUM_SLICES).round().astype(int)

        volume_slices = []

        # Pre-load all necessary slices to avoid re-reading for 2.5D stacking?
        # Optimization: Only read unique indices needed for 2.5D (idx-1, idx, idx+1)
        # But for simplicity and memory, we read on demand or cache in memory map.
        # Given 64 slices, we need at most 64*3 reads.

        # To optimize, we can map original index to loaded image
        loaded_images = {}

        def get_slice(i):
            # Clamp index
            i = max(0, min(num_files - 1, i))
            if i not in loaded_images:
                loaded_images[i] = load_dicom_slice(slice_files[i], Config.IMAGE_SIZE)
            return loaded_images[i]

        for i in indices:
            # 2.5D Stacking: (i-1, i, i+1)
            img_prev = get_slice(i - 1)
            img_curr = get_slice(i)
            img_next = get_slice(i + 1)

            # Stack along channel axis
            stack = np.stack([img_prev, img_curr, img_next], axis=-1)  # (H, W, 3)
            volume_slices.append(stack)

        volume = np.stack(volume_slices, axis=0)  # (64, H, W, 3)

        # Save to cache
        if self.load_cached_data:
            np.save(cache_path, volume)

        return volume

    def _apply_consistent_augmentations(self, volume):
        """
        Applies the same geometric transformation to all slices in the volume.
        volume: (D, H, W, C) uint8
        """
        # Create a replayable composition
        # We assume self.transform is an Albumentations ReplayCompose or similar
        # If it's a standard Compose, we can't easily replay without ReplayCompose.
        # However, we can hack it by using ReplayCompose wrapper if not provided,
        # but here we assume the user provides a compatible transform or we handle it.

        # If the provided transform is not ReplayCompose, we can't guarantee consistency easily
        # unless we manually seed.
        # Strategy: Use ReplayCompose on the first slice, then replay on others.

        # Check if volume is empty
        if volume.shape[0] == 0:
            return torch.zeros(
                (Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            )

        # 1. Augment first slice to get parameters
        first_slice = volume[0]  # (H, W, 3)

        # Ensure transform is ReplayCompose-capable or we wrap it?
        # For simplicity, we assume self.transform is a ReplayCompose instance.
        # If it is just Compose, we can convert it or use it as is (random per slice - bad).
        # We will enforce ReplayCompose in get_dataloaders.

        data = self.transform(image=first_slice)
        augmented_slices = [data["image"]]
        replay_params = data.get("replay", None)

        # 2. Replay on remaining slices
        for i in range(1, len(volume)):
            img = volume[i]
            if replay_params:
                res = A.ReplayCompose.replay(replay_params, image=img)
                augmented_slices.append(res["image"])
            else:
                # Fallback if no replay params (e.g. only normalization)
                res = self.transform(image=img)
                augmented_slices.append(res["image"])

        # Stack back to tensor
        # Albumentations ToTensorV2 converts to (C, H, W) tensor
        # So augmented_slices is a list of (C, H, W) tensors
        volume_tensor = torch.stack(augmented_slices, dim=0)  # (D, C, H, W)

        return volume_tensor

    def _get_labels(self, row):
        if self.subset == "test":
            # Return dummy labels for test set
            return torch.zeros(Config.NUM_CLASSES + 1, dtype=torch.float32)

        # Labels: C1, C2, C3, C4, C5, C6, C7, patient_overall
        # Note: Config.NUM_CLASSES is 7. We append patient_overall.
        label_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
        labels = row[label_cols].values.astype(np.float32)
        return torch.tensor(labels)


def get_dataloaders(debug_sample_size=None):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debugging: Subsample
    if debug_sample_size is not None:
        train_df = train_df.iloc[:debug_sample_size]
        val_df = val_df.iloc[:debug_sample_size]
        # Keep test full or subsample? Usually keep test full for submission check,
        # but for speed we might subsample.
        # test_df = test_df.iloc[:debug_sample_size]

    # 2. Define Transforms
    # We use ReplayCompose for volumetric consistency
    train_transform = A.ReplayCompose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.5
            ),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.ReplayCompose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 3. Create Datasets
    train_dataset = RSNADataset(
        train_df, subset="train", load_cached_data=True, transform=train_transform
    )
    val_dataset = RSNADataset(
        val_df, subset="val", load_cached_data=True, transform=val_transform
    )
    test_dataset = RSNADataset(
        test_df, subset="test", load_cached_data=True, transform=val_transform
    )

    # 4. Create DataLoaders
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
