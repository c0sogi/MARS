import os
import re
import glob
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def extract_slice_id(filename):
    """Extracts the integer slice ID from a filename like 'Image-123.dcm'."""
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def get_sorted_files_and_ids(folder_path):
    """
    Returns a sorted list of (slice_id, filename) tuples for a given folder.
    """
    if not os.path.exists(folder_path):
        return []

    files = os.listdir(folder_path)
    # Filter for likely DICOM files
    files = [f for f in files if "dcm" in f.lower()]

    # Parse IDs
    file_data = []
    for f in files:
        sid = extract_slice_id(f)
        if sid != -1:
            file_data.append((sid, f))

    # Sort by ID
    file_data.sort(key=lambda x: x[0])
    return file_data


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class BrainTumorDataset(Dataset):
    def __init__(self, metadata_df, is_train=False, load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing subject info and paths.
            is_train (bool): Whether this is a training set (enables augmentation).
            load_cached_data (bool): Whether to load/save anchor indices from cache.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.is_train = is_train
        self.load_cached_data = load_cached_data

        # Cache path for ROI/Anchor indices
        self.cache_path = os.path.join(config.CACHE_DIR, "roi_cache.parquet")

        # Precompute or load anchor slice IDs
        self.anchors = self._initialize_anchors()

    def _initialize_anchors(self):
        """
        Manages the caching logic for anchor slice selection.
        """
        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            try:
                cache_df = pd.read_parquet(self.cache_path)
                # Convert to dictionary for fast lookup: BraTS21ID -> Anchor Slice ID
                anchor_map = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_id"]))

                # Verify all current subjects are in cache
                all_present = True
                for pid in self.metadata["BraTS21ID"].unique():
                    if pid not in anchor_map:
                        all_present = False
                        break

                if all_present:
                    return anchor_map
            except Exception:
                pass  # Fallback to recomputing

        # 2. Compute from scratch
        anchor_map = {}
        print(f"Computing anchor slices for {len(self.metadata)} subjects...")

        for idx, row in self.metadata.iterrows():
            subject_id = row["BraTS21ID"]
            flair_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])

            anchor_id = self._find_anchor_slice(flair_path)
            anchor_map[subject_id] = anchor_id

        # 3. Save to cache
        try:
            cache_data = [
                {"BraTS21ID": k, "anchor_id": v} for k, v in anchor_map.items()
            ]
            pd.DataFrame(cache_data).to_parquet(self.cache_path)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return anchor_map

    def _find_anchor_slice(self, flair_dir):
        """
        Scans FLAIR directory to find the slice with max intensity sum
        within the 15-85% depth range.
        """
        files = get_sorted_files_and_ids(flair_dir)
        if not files:
            return 0  # Fallback

        num_files = len(files)
        start_idx = int(num_files * config.ROI_DEPTH_RANGE[0])
        end_idx = int(num_files * config.ROI_DEPTH_RANGE[1])

        # Ensure valid range
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_files

        best_sum = -1.0
        best_id = files[num_files // 2][0]  # Default to middle

        # Iterate through the range
        for i in range(start_idx, end_idx):
            sid, fname = files[i]
            path = os.path.join(flair_dir, fname)

            # Use utils to load (robust fallback)
            img = utils.load_dicom_image(path)

            current_sum = np.sum(img)
            if current_sum > best_sum:
                best_sum = current_sum
                best_id = sid

        return best_id

    def _load_volume_slice(self, dir_path, target_id):
        """
        Loads a specific slice ID from a modality directory.
        Handles edge clamping if the ID is out of bounds.
        """
        files = get_sorted_files_and_ids(dir_path)
        if not files:
            return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

        # Extract available IDs
        available_ids = [f[0] for f in files]
        min_id = available_ids[0]
        max_id = available_ids[-1]

        # Edge Clamping
        if target_id < min_id:
            lookup_id = min_id
        elif target_id > max_id:
            lookup_id = max_id
        else:
            # Try to find exact match, else nearest
            if target_id in available_ids:
                lookup_id = target_id
            else:
                # Find nearest ID
                lookup_id = min(available_ids, key=lambda x: abs(x - target_id))

        # Find filename for lookup_id
        fname = next(f[1] for f in files if f[0] == lookup_id)
        full_path = os.path.join(dir_path, fname)

        return utils.load_dicom_image(full_path)

    def _augment(self, volume):
        """
        Applies geometric augmentations to the (H, W, C) volume.
        volume: numpy array (224, 224, 20)
        """
        # 1. Random Rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(
                -config.AUG_ROTATION_RANGE, config.AUG_ROTATION_RANGE
            )
            h, w = volume.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply to all channels
            # cv2.warpAffine works on multiple channels if they are stacked correctly,
            # but sometimes behaves unexpectedly with >4 channels depending on version.
            # To be safe and explicit, we loop or use specific multi-channel support.
            # However, looping is slow. Let's try processing as a block if possible,
            # or loop if necessary. EfficientNet input is 20 channels.

            augmented_channels = []
            for c in range(volume.shape[2]):
                # Use REFLECT padding as requested
                aug_slice = cv2.warpAffine(
                    volume[:, :, c],
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                augmented_channels.append(aug_slice)
            volume = np.stack(augmented_channels, axis=-1)

        # 2. Horizontal Flip
        if np.random.rand() < 0.5:
            volume = np.flip(volume, axis=1)  # Flip columns (width)

        # 3. Vertical Flip
        if np.random.rand() < 0.5:
            volume = np.flip(volume, axis=0)  # Flip rows (height)

        return volume

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        subject_id = row["BraTS21ID"]

        # 1. Get Anchor
        anchor_id = self.anchors.get(subject_id, 0)

        # 2. Define Target Slice IDs (Stride 2: -4, -2, 0, +2, +4)
        offsets = [-4, -2, 0, 2, 4]
        target_ids = [anchor_id + o for o in offsets]

        # 3. Load Data
        # Structure: 4 Modalities * 5 Slices = 20 Channels
        # Order: FLAIR(5), T1w(5), T1wCE(5), T2w(5)

        channels = []

        for mod in config.MODALITIES:
            mod_path_rel = row[f"path_{mod}"]
            mod_dir = os.path.join(config.INPUT_DIR, mod_path_rel)

            for tid in target_ids:
                # Load
                img = self._load_volume_slice(mod_dir, tid)

                # Normalize (Independent Per-Channel Min-Max)
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)

                channels.append(img)

        # Stack to (H, W, 20)
        volume = np.stack(channels, axis=-1)

        # 4. Augmentation (Train only)
        if self.is_train:
            volume = self._augment(volume)

        # 5. Convert to Tensor (C, H, W)
        # Current: (H, W, C) -> Transpose to (C, H, W)
        volume = np.transpose(volume, (2, 0, 1))
        tensor = torch.from_numpy(volume.copy()).float()

        # 6. Return
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return tensor, target
        else:
            return tensor


# -----------------------------------------------------------------------------
# Data Loader Builder
# -----------------------------------------------------------------------------


def get_dataloader(
    metadata_path,
    batch_size=config.BATCH_SIZE,
    is_train=False,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Factory function to create a DataLoader.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    dataset = BrainTumorDataset(
        metadata_df=df, is_train=is_train, load_cached_data=load_cached_data
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
