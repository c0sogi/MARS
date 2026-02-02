import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
import albumentations as A
import json
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_loader", "data_loader.log")


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., 1.dcm, 2.dcm, 10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def get_study_file_map(metadata_df, cache_dir, load_cached_data=True, phase="train"):
    """
    Creates or loads a map of StudyInstanceUID -> List of sorted file paths.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'StudyInstanceUID' and 'image_path'.
        cache_dir (str): Directory to store the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.
        phase (str): The dataset phase (train/val/test) for cache scoping.

    Returns:
        dict: Mapping of StudyInstanceUID to list of absolute file paths.
    """
    cache_path = os.path.join(cache_dir, f"{phase}_paths_cache.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            logger.info(f"Loading file paths from cache: {cache_path}")
            # We store as a long format DataFrame: StudyInstanceUID, RelativePath, SliceNum
            # But for speed, let's just store the serialized lists or reconstruct.
            # Storing lists in parquet can be tricky, so we'll store a long DF.
            df_cache = pd.read_parquet(cache_path)

            # Group back into dictionary
            # Assuming columns: StudyInstanceUID, filename
            # We need to reconstruct the full paths.
            # To save space, the cache might only have filenames.

            # Let's rebuild the dictionary
            study_file_map = {}

            # Optimized grouping
            grouped = df_cache.groupby("StudyInstanceUID")["filename"].apply(list)

            # We need to rejoin with the base path from metadata
            # Create a lookup for study -> image_dir
            study_to_dir = pd.Series(
                metadata_df["image_path"].values, index=metadata_df["StudyInstanceUID"]
            ).to_dict()

            for study_id, filenames in grouped.items():
                if study_id in study_to_dir:
                    base_dir = os.path.join(Config.INPUT_DIR, study_to_dir[study_id])
                    # Ensure filenames are sorted naturally (though they should be saved sorted)
                    # The cache load order might not be guaranteed, so we sort again to be safe
                    filenames.sort(key=natural_sort_key)
                    study_file_map[study_id] = [
                        os.path.join(base_dir, f) for f in filenames
                    ]

            return study_file_map

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info("Scanning directories for DICOM files...")
    study_file_map = {}
    cache_rows = []

    unique_studies = metadata_df["StudyInstanceUID"].unique()
    study_to_path = pd.Series(
        metadata_df["image_path"].values, index=metadata_df["StudyInstanceUID"]
    ).to_dict()

    for study_id in unique_studies:
        rel_path = study_to_path[study_id]
        full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_dir_path):
            # List all .dcm files
            files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]
            # Sort naturally
            files.sort(key=natural_sort_key)

            full_paths = [os.path.join(full_dir_path, f) for f in files]
            study_file_map[study_id] = full_paths

            # Add to cache list
            for f in files:
                cache_rows.append({"StudyInstanceUID": study_id, "filename": f})
        else:
            logger.warning(f"Directory not found: {full_dir_path}")
            study_file_map[study_id] = []

    # 3. Save to cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        df_cache = pd.DataFrame(cache_rows)
        # Ensure slice number sorting is implicit by insertion order or filename
        df_cache.to_parquet(cache_path, index=False)
        logger.info(f"Saved file paths to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return study_file_map


def load_dicom_slice(path, size=(256, 256)):
    """
    Loads a DICOM file, applies bone windowing, resizes, and normalizes.

    Args:
        path (str): Path to the DICOM file.
        size (tuple): Target size (H, W).

    Returns:
        np.ndarray: Processed image of shape (H, W) with values in [0, 1].
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Apply Rescale Slope/Intercept if present
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Bone Windowing
        # Center (Level) = 1000, Width = 2000
        # Range: [1000 - 1000, 1000 + 1000] -> [0, 2000]
        window_center = 1000
        window_width = 2000

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        if img_max != img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

    except Exception as e:
        # Fallback for corrupt files
        img = np.zeros(size, dtype=np.float32)

    # Resize
    if img.shape != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    return img


class RSNADataset(Dataset):
    """
    Dataset for RSNA Cervical Spine Fracture Detection.
    Loads 2.5D stacks of slices for a full study sequence.
    """

    def __init__(
        self, metadata_df, transforms=None, load_cached_data=True, phase="train"
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing study IDs and labels.
            transforms (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to use cached file paths.
            phase (str): 'train', 'val', or 'test'.
        """
        self.metadata_df = metadata_df
        self.transforms = transforms
        self.phase = phase
        self.seq_len = Config.SEQ_LEN
        self.image_size = Config.IMAGE_SIZE

        # Get file map
        self.study_file_map = get_study_file_map(
            metadata_df,
            Config.WORKING_DIR,
            load_cached_data=load_cached_data,
            phase=self.phase,
        )

        # Filter out studies with no images
        valid_studies = []
        for uid in self.metadata_df["StudyInstanceUID"].unique():
            if uid in self.study_file_map and len(self.study_file_map[uid]) > 0:
                valid_studies.append(uid)

        if len(valid_studies) < len(self.metadata_df):
            logger.warning(
                f"Filtered out {len(self.metadata_df) - len(valid_studies)} studies due to missing images."
            )
            self.metadata_df = self.metadata_df[
                self.metadata_df["StudyInstanceUID"].isin(valid_studies)
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Get all file paths for this study
        files = self.study_file_map[study_id]
        num_files = len(files)

        # Uniform Sampling of indices
        # We want exactly SEQ_LEN indices
        if num_files >= self.seq_len:
            indices = np.linspace(0, num_files - 1, self.seq_len).astype(int)
        else:
            # If fewer slices than sequence length, interpolate indices
            # Or simpler: repeat slices.
            # linspace handles the interpolation of indices naturally,
            # but we will have duplicates.
            indices = np.linspace(0, num_files - 1, self.seq_len).astype(int)

        # Load Images
        # We need to construct 2.5D input: channels = [z-1, z, z+1]
        # Result shape: (Seq_Len, 3, H, W)

        # To optimize augmentation, we can stack all channels: (H, W, Seq_Len * 3)
        # Apply 2D augmentation, then reshape back.

        stacked_images = []

        for i in indices:
            # Determine neighbor indices with clamping
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_files - 1, i + 1)

            # Load slices
            # Note: This involves 3 file reads per sequence step.
            # Optimization: Cache loaded slices in a dict for this iteration if there's overlap?
            # With stride > 1, overlap is rare. With dense sampling, overlap happens.
            # Given SEQ_LEN=96 and typical study size ~300, stride is ~3. Overlap is minimal.

            img_prev = load_dicom_slice(files[idx_prev], self.image_size)
            img_curr = load_dicom_slice(files[idx_curr], self.image_size)
            img_next = load_dicom_slice(files[idx_next], self.image_size)

            # Stack to (H, W, 3)
            slice_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
            stacked_images.append(slice_25d)

        # Concatenate along channel dimension for volumetric augmentation
        # Shape: (H, W, Seq_Len * 3)
        volume = np.concatenate(stacked_images, axis=-1)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=volume)
            volume = augmented["image"]

        # Reshape back to (Seq_Len, 3, H, W) and convert to Tensor
        # Current volume: (H, W, Seq_Len * 3)
        # Transpose to (Seq_Len * 3, H, W)
        volume = volume.transpose(2, 0, 1)

        # Reshape to (Seq_Len, 3, H, W)
        volume = volume.reshape(self.seq_len, 3, self.image_size[0], self.image_size[1])

        # Convert to torch tensor
        images_tensor = torch.from_numpy(volume).float()

        # Get Targets
        if self.phase != "test":
            labels = row[Config.TARGET_COLUMNS].values.astype(np.float32)
            targets_tensor = torch.tensor(labels)
            return images_tensor, targets_tensor
        else:
            # For test set, we might not have targets, return study_id for submission mapping
            return images_tensor, study_id


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specific phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # RandomBrightnessContrast is tricky on normalized [0,1] bone window data
                # but can help robustness.
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.2
                ),
            ]
        )
    else:
        return None


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    load_cached_data=True,
    debug=False,
):
    """
    Factory function to create DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)

    # Debug mode: subsample
    if debug:
        train_df = train_df.iloc[:10]
        val_df = val_df.iloc[:10]
        logger.info("DEBUG MODE: Reduced dataset size to 10 samples.")

    # Datasets
    train_dataset = RSNADataset(
        train_df,
        transforms=get_transforms("train"),
        load_cached_data=load_cached_data,
        phase="train",
    )

    val_dataset = RSNADataset(
        val_df,
        transforms=get_transforms("val"),
        load_cached_data=load_cached_data,
        phase="val",
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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(
    test_metadata_path=Config.TEST_METADATA_PATH, load_cached_data=True
):
    """
    Factory function for Test DataLoader.
    """
    test_df = pd.read_csv(test_metadata_path)

    test_dataset = RSNADataset(
        test_df,
        transforms=get_transforms("test"),
        load_cached_data=load_cached_data,
        phase="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
