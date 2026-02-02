import os
import glob
import numpy as np
import pandas as pd
import torch
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

logger = get_logger(__name__)


def get_bbox_data(load_cached_data=True):
    """
    Loads and processes bounding box data with caching.
    Returns a dictionary mapping StudyInstanceUID to a set of fractured slice numbers.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "bbox_cache.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached bounding box data from {cache_path}")
        bbox_df = pd.read_parquet(cache_path)
    else:
        logger.info("Processing bounding box data from scratch...")
        if not os.path.exists(Config.BOUNDING_BOX_PATH):
            logger.warning(
                f"Bounding box file not found at {Config.BOUNDING_BOX_PATH}. Returning empty map."
            )
            return {}

        raw_df = pd.read_csv(Config.BOUNDING_BOX_PATH)
        # Keep only necessary columns
        bbox_df = raw_df[["StudyInstanceUID", "slice_number"]].copy()

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        bbox_df.to_parquet(cache_path, index=False)
        logger.info(f"Saved bounding box cache to {cache_path}")

    # Convert to dictionary for fast lookup: {uid: {slice_num, slice_num, ...}}
    bbox_map = bbox_df.groupby("StudyInstanceUID")["slice_number"].apply(set).to_dict()
    return bbox_map


def read_dicom_normalized(path):
    """
    Reads a DICOM file, applies bone windowing, and normalizes to [0, 1].
    """
    try:
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        slope = getattr(dcm, "RescaleSlope", 1)
        image = image * slope + intercept

        # Apply Bone Window (Level 300, Width 2000)
        center = 300
        width = 2000
        lower = center - (width / 2)
        upper = center + (width / 2)

        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)

        return image.astype(np.float32)
    except Exception as e:
        # Return a blank image if reading fails
        # Assuming 512x512 default, but will be resized later anyway
        return np.zeros((512, 512), dtype=np.float32)


class CervicalSpineDataset(Dataset):
    def __init__(self, metadata_df, bbox_map, transform=None, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing StudyInstanceUID and paths.
            bbox_map (dict): Dictionary mapping UID to set of fractured slice numbers.
            transform (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df
        self.bbox_map = bbox_map
        self.transform = transform
        self.mode = mode

        # Pre-compute file lists could be slow if done here for all,
        # but we do it lazily or rely on OS caching.
        # Given 1.5k studies, we can store paths in the dataframe if not already there.
        # The metadata df passed here already has 'image_path'.

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]
        rel_path = row["image_path"]

        # Construct full path
        study_dir = os.path.join(Config.INPUT_ROOT, rel_path)

        # Get all DICOM files and sort them by slice number (filename integer)
        # Filenames are like '1.dcm', '10.dcm', etc.
        try:
            files = glob.glob(os.path.join(study_dir, "*.dcm"))
            # Sort by the integer value of the filename (slice number)
            files = sorted(
                files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
        except Exception:
            files = []

        num_slices = len(files)
        seq_len = Config.SEQ_LEN

        # --- Sampling Indices ---
        if num_slices == 0:
            # Handle empty directory edge case
            indices = np.zeros(seq_len, dtype=int)
            files = [None]  # Placeholder
        else:
            # Uniformly sample indices
            indices = np.linspace(0, num_slices - 1, seq_len).astype(int)

        # --- Load Images (2.5D Stacking) ---
        # We need to stack (z-1, z, z+1) for each sampled index.
        # Total images to load = seq_len * 3 (but many overlap).
        # Optimization: Load unique needed slices, then assemble.

        # Identify all unique slice indices needed
        needed_indices = set()
        for i in indices:
            needed_indices.add(i)
            needed_indices.add(max(0, i - 1))
            needed_indices.add(min(num_slices - 1, i + 1))

        # Load unique slices into memory
        loaded_slices = {}
        for i in needed_indices:
            if files[0] is None:
                loaded_slices[i] = np.zeros(Config.IMAGE_SIZE, dtype=np.float32)
            else:
                loaded_slices[i] = read_dicom_normalized(files[i])

        # Assemble the sequence
        # Shape: (Seq_Len, H, W, 3) -> effectively (H, W, Seq_Len * 3) for Albumentations
        stacked_channels = []

        for i in indices:
            # Boundary handling: clamp to [0, num_slices-1]
            prev_idx = max(0, i - 1)
            next_idx = min(num_slices - 1, i + 1)

            img_prev = loaded_slices[prev_idx]
            img_curr = loaded_slices[i]
            img_next = loaded_slices[next_idx]

            # Stack along channel dim
            # Each is (H, W), stack -> (H, W, 3)
            # We append to list to flatten later
            stacked_channels.extend([img_prev, img_curr, img_next])

        # Convert to numpy: (H, W, Seq_Len * 3)
        # Note: Albumentations expects (H, W, C)
        volume = np.dstack(stacked_channels)

        # --- Augmentation ---
        if self.transform:
            augmented = self.transform(image=volume)
            volume = augmented["image"]  # Returns Tensor (C, H, W) due to ToTensorV2
        else:
            # Fallback if no transform (shouldn't happen based on get_loaders)
            volume = torch.from_numpy(volume.transpose(2, 0, 1))

        # Reshape to (Seq_Len, 3, H, W)
        # Current shape: (Seq_Len * 3, H, W)
        c, h, w = volume.shape
        volume = volume.view(seq_len, 3, h, w)

        # --- Targets ---
        # 1. Study Targets
        if self.mode != "test":
            labels = row[Config.TARGET_COLUMNS].values.astype(np.float32)
            study_targets = torch.tensor(labels)
        else:
            study_targets = torch.zeros(8)  # Dummy

        # 2. Slice Targets (Auxiliary)
        slice_targets = np.zeros(seq_len, dtype=np.float32)
        has_bbox = 0.0

        if self.mode != "test":
            # Check if this UID has bbox data
            if uid in self.bbox_map:
                has_bbox = 1.0
                fractured_slices = self.bbox_map[uid]

                for idx, slice_idx in enumerate(indices):
                    # Map sampled index back to slice number
                    # files[slice_idx] is the path, basename is '100.dcm'
                    if files[0] is not None:
                        fname = os.path.basename(files[slice_idx])
                        slice_num = int(os.path.splitext(fname)[0])

                        if slice_num in fractured_slices:
                            slice_targets[idx] = 1.0

        slice_targets = torch.tensor(slice_targets)
        slice_mask = torch.tensor([has_bbox], dtype=torch.float32)

        return {
            "images": volume,  # (Seq, 3, H, W)
            "study_targets": study_targets,  # (8,)
            "slice_targets": slice_targets,  # (Seq,)
            "slice_mask": slice_mask,  # (1,)
            "row_id": uid,  # Helper for submission
        }


def get_transforms(mode="train"):
    """
    Returns albumentations transforms.
    We apply geometric transforms to the entire volume (stacked channels) to ensure consistency.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                # A.HorizontalFlip(p=0.5), # Optional, safe for vertical spine levels
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), ToTensorV2()]
        )


def get_loaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Load BBox Map
    bbox_map = get_bbox_data(load_cached_data=load_cached_data)

    # 3. Create Datasets
    train_ds = CervicalSpineDataset(
        train_df, bbox_map, transform=get_transforms("train"), mode="train"
    )
    val_ds = CervicalSpineDataset(
        val_df, bbox_map, transform=get_transforms("val"), mode="val"
    )
    test_ds = CervicalSpineDataset(
        test_df, bbox_map, transform=get_transforms("test"), mode="test"
    )

    # 4. Create DataLoaders
    # Use drop_last=True for train to maintain batch shape consistency if needed
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
