import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import from provided library files
from library.config import Config
from library.utils import load_dicom_slice, get_brain_roi_depth


def prepare_roi_cache(metadata_df, cache_name, load_cached_data=True):
    """
    Computes or loads the Region of Interest (ROI) start and end indices for each subject's modalities.

    Args:
        metadata_df (pd.DataFrame): Dataframe containing subject paths.
        cache_name (str): Unique name for the cache file (e.g., 'train_roi', 'val_roi').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping BraTS21ID to a dictionary of modality ROIs.
              Format: {subject_id: {'FLAIR': (start, end), 'T1wCE': ..., 'T2w': ...}}
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            roi_data = np.load(cache_path, allow_pickle=True).item()
            return roi_data
        except Exception:
            pass  # Fallback to computation if load fails

    # 2. Compute from scratch
    roi_data = {}

    # Iterate over dataframe
    # We use the dataframe index or rows. metadata_df is expected to have BraTS21ID and path columns.
    for _, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]
        subject_rois = {}

        for modality in Config.MODALITIES:
            # Column name in metadata is lowercase usually, e.g., 'flair_path'
            # Config.MODALITIES are ["FLAIR", "T1wCE", "T2w"]
            col_name = f"{modality.lower()}_path"
            rel_path = row[col_name]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            start_idx, end_idx, _ = get_brain_roi_depth(full_path)
            subject_rois[modality] = (start_idx, end_idx)

        roi_data[subject_id] = subject_rois

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, roi_data)

    return roi_data


class BraTSDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        transform=None,
        is_train=True,
        load_cached_data=True,
        cache_name="dataset_cache",
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            transform (albumentations.Compose): Augmentation pipeline.
            is_train (bool): Whether this is a training dataset.
            load_cached_data (bool): Whether to use cached ROI calculations.
            cache_name (str): Identifier for the cache file.
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

        # Pre-calculate or load ROI indices for Anatomical Anchoring
        # This prevents re-scanning thousands of files during training
        self.roi_cache = prepare_roi_cache(
            self.metadata_df, cache_name, load_cached_data=load_cached_data
        )

        # We expand the dataset: 3 samples per subject (one for each depth)
        self.slab_depths = Config.SLAB_DEPTHS
        self.samples_per_subject = len(self.slab_depths)

    def __len__(self):
        return len(self.metadata_df) * self.samples_per_subject

    def __getitem__(self, index):
        # Map linear index to subject and slab depth
        subject_idx = index // self.samples_per_subject
        depth_idx = index % self.samples_per_subject

        row = self.metadata_df.iloc[subject_idx]
        subject_id = row["BraTS21ID"]
        depth_ratio = self.slab_depths[depth_idx]

        # Get label if available
        target = -1.0
        if "MGMT_value" in row:
            target = float(row["MGMT_value"])

        # Retrieve ROI data for this subject
        subject_rois = self.roi_cache.get(subject_id)

        channels = []

        # Iterate through modalities in fixed order: FLAIR, T1wCE, T2w
        for modality in Config.MODALITIES:
            # 1. Determine Anatomical Anchor
            if subject_rois:
                start_roi, end_roi = subject_rois[modality]
            else:
                # Fallback if cache lookup fails (should not happen)
                start_roi, end_roi = 0, 0

            roi_len = end_roi - start_roi
            if roi_len < 1:
                # Fallback for empty/bad folders: assume middle of whatever files exist
                # We need to list files to know length, but to be fast we might just assume 0
                # Ideally get_brain_roi_depth handles this.
                center_idx = start_roi
            else:
                center_idx = int(start_roi + (roi_len * depth_ratio))

            # 2. Define Slab (z-1, z, z+1)
            # We need to map this index to the actual file list.
            # However, get_brain_roi_depth returns indices relative to the sorted file list.
            # We need to construct the file paths.

            col_name = f"{modality.lower()}_path"
            modality_dir = os.path.join(Config.INPUT_DIR, row[col_name])

            # We need the file list to access by index.
            # Optimization: We don't want to listdir every time.
            # However, storing all file lists in memory is heavy.
            # Compromise: Listdir is cached by OS usually, but let's do it safely.
            # Since we need specific files, we must list them.
            # To avoid listing every time, we could have cached file lists, but ROI cache is indices.
            # We will list here. It's an I/O cost but necessary without massive RAM usage.

            try:
                files = sorted(
                    [f for f in os.listdir(modality_dir) if f.endswith(".dcm")],
                    key=lambda x: int(x.split("-")[1].split(".")[0]) if "-" in x else 0,
                )
            except FileNotFoundError:
                files = []

            num_files = len(files)
            if num_files == 0:
                # Create blank slices if missing
                slab_slices = [
                    np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
                    for _ in range(Config.NUM_SLICES_PER_SLAB)
                ]
            else:
                # Clamp center index
                center_idx = max(0, min(center_idx, num_files - 1))

                # Indices for the slab
                half_window = Config.NUM_SLICES_PER_SLAB // 2
                slice_indices = range(
                    center_idx - half_window, center_idx + half_window + 1
                )

                slab_slices = []
                for idx in slice_indices:
                    # Clamp neighbor indices
                    idx = max(0, min(idx, num_files - 1))
                    file_path = os.path.join(modality_dir, files[idx])

                    try:
                        img = load_dicom_slice(file_path)

                        # Resize immediately to save memory/compute
                        img = cv2.resize(
                            img,
                            (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        # Independent Channel Min-Max Normalization
                        # Avoid division by zero
                        min_val = np.min(img)
                        max_val = np.max(img)
                        if max_val - min_val > 0:
                            img = (img - min_val) / (max_val - min_val)
                        else:
                            img = np.zeros_like(img)

                    except Exception:
                        img = np.zeros(
                            (Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
                        )

                    slab_slices.append(img)

            channels.extend(slab_slices)

        # Stack all channels: (H, W, 9)
        # 3 slices * 3 modalities = 9 channels
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, torch.tensor(target, dtype=torch.float32)


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE), ToTensorV2()])
