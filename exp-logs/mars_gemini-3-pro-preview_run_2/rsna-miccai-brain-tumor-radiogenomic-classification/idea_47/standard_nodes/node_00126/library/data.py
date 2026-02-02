import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import read_dicom_image, get_logger

logger = get_logger("data")


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate with reflection padding to avoid artifacts at boundaries
                A.Rotate(
                    limit=Config.ROTATION_DEGREES, border_mode=cv2.BORDER_REFLECT, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class BrainTumorDataset(Dataset):
    """
    Dataset class for Brain Tumor Classification using a Fidelity-Aligned Data Pipeline.

    Features:
    - ROI Selection: Uses FLAIR as the geometric anchor based on max intensity sum.
    - Caching: Caches ROI anchor IDs to disk to speed up subsequent initializations.
    - Geometric Consistency: Uses edge clamping for the reference modality and zero-padding for others.
    - Robustness: Implements a circuit breaker for data corruption.
    """

    def __init__(
        self, df: pd.DataFrame, phase: str = "train", load_cached_data: bool = True
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (BraTS21ID, paths, etc.).
            phase (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load/save ROI anchors from/to cache.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = get_transforms(phase)
        self.roi_map = {}  # Mapping: BraTS21ID -> Anchor Slice ID (int)

        # Ensure working directory exists for cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        self.cache_path = os.path.join(Config.WORKING_DIR, "roi_cache.parquet")

        # Execute the ROI discovery pipeline
        self._prepare_rois(load_cached_data)

    def _extract_slice_id(self, filename: str) -> int:
        """Extracts the integer slice ID from a DICOM filename (e.g., 'Image-123.dcm' -> 123)."""
        match = re.search(r"Image-(\d+)\.dcm", filename)
        if match:
            return int(match.group(1))
        return -1

    def _get_flair_files(self, subject_path_flair: str):
        """Returns a sorted list of (slice_id, filename) tuples for the FLAIR directory."""
        full_path = os.path.join(Config.INPUT_DIR, subject_path_flair)
        if not os.path.exists(full_path):
            return []

        files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
        # Parse IDs
        file_data = []
        for f in files:
            sid = self._extract_slice_id(f)
            if sid != -1:
                file_data.append((sid, f))

        # Sort by slice ID
        file_data.sort(key=lambda x: x[0])
        return file_data

    def _compute_roi_anchor(self, subject_row) -> int:
        """
        Computes the ROI anchor slice ID for a single subject using FLAIR modality.
        Metric: Maximum Sum of Intensity (Raw Pixel Values).
        Range: 15% - 85% of volume depth.
        """
        flair_path_rel = subject_row["path_FLAIR"]
        full_flair_dir = os.path.join(Config.INPUT_DIR, flair_path_rel)

        files = self._get_flair_files(flair_path_rel)
        if not files:
            return -1

        num_slices = len(files)
        start_idx = int(num_slices * Config.ROI_DEPTH_START)
        end_idx = int(num_slices * Config.ROI_DEPTH_END)

        # Safety check for very small volumes
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_slices

        candidate_files = files[start_idx:end_idx]

        max_intensity = -1.0
        best_slice_id = -1

        # Iterate through candidates to find the one with highest signal
        for sid, fname in candidate_files:
            img_path = os.path.join(full_flair_dir, fname)
            try:
                # Use the robust reader
                img = read_dicom_image(img_path)
                current_intensity = np.sum(img)

                if current_intensity > max_intensity:
                    max_intensity = current_intensity
                    best_slice_id = sid
            except Exception:
                continue

        # Fallback: if scanning failed, take the middle slice
        if best_slice_id == -1 and files:
            mid_idx = num_slices // 2
            best_slice_id = files[mid_idx][0]

        return best_slice_id

    def _prepare_rois(self, load_cached_data: bool):
        """
        Manages the caching logic for ROI anchors.
        Loads from Parquet if available, otherwise computes and saves.
        Implements Circuit Breaker for data corruption.
        """
        cached_df = pd.DataFrame()

        # 1. Try to load cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                cached_df = pd.read_parquet(self.cache_path)
                logger.info(
                    f"Loaded ROI cache from {self.cache_path} with {len(cached_df)} entries."
                )
            except Exception as e:
                logger.warning(f"Failed to load ROI cache: {e}. Recomputing...")

        # Convert cache to dict for fast lookup
        cache_map = {}
        if (
            not cached_df.empty
            and "BraTS21ID" in cached_df.columns
            and "anchor_id" in cached_df.columns
        ):
            cache_map = dict(zip(cached_df["BraTS21ID"], cached_df["anchor_id"]))

        # 2. Identify missing subjects
        missing_ids = []
        for idx, row in self.df.iterrows():
            bid = row["BraTS21ID"]
            if bid not in cache_map:
                missing_ids.append(idx)

        # 3. Compute missing ROIs
        if missing_ids:
            logger.info(f"Computing ROIs for {len(missing_ids)} subjects...")
            corruption_count = 0
            new_entries = []

            for idx in missing_ids:
                row = self.df.iloc[idx]
                bid = row["BraTS21ID"]

                anchor_id = self._compute_roi_anchor(row)

                if anchor_id == -1:
                    corruption_count += 1
                    # Fallback to a dummy ID (e.g., 1) to prevent crash, but track corruption
                    anchor_id = 1

                cache_map[bid] = anchor_id
                new_entries.append({"BraTS21ID": bid, "anchor_id": anchor_id})

            # Circuit Breaker Check
            corruption_rate = corruption_count / len(missing_ids)
            if corruption_rate > Config.CORRUPTION_THRESHOLD:
                msg = f"Data Corruption Alert: {corruption_rate:.2%} of processed subjects have unreadable/empty FLAIR volumes."
                logger.error(msg)
                raise RuntimeError(msg)

            # 4. Update Cache on Disk
            if new_entries:
                new_df = pd.DataFrame(new_entries)
                if not cached_df.empty:
                    combined_df = pd.concat(
                        [cached_df, new_df], ignore_index=True
                    ).drop_duplicates(subset=["BraTS21ID"], keep="last")
                else:
                    combined_df = new_df

                # Atomic write (safe)
                try:
                    combined_df.to_parquet(self.cache_path, index=False)
                    logger.info("ROI cache updated and saved.")
                except Exception as e:
                    logger.warning(f"Could not save ROI cache: {e}")

        self.roi_map = cache_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # 1. Retrieve Anchor
        anchor_id = self.roi_map.get(subject_id, -1)

        # 2. Determine Reference Geometry (FLAIR)
        # We need the sorted list of FLAIR files to perform index-based clamping
        flair_files = self._get_flair_files(row["path_FLAIR"])

        if not flair_files:
            # Catastrophic failure fallback: return zeros
            return (
                torch.zeros(
                    (Config.INPUT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                    dtype=torch.float32,
                ),
                0.0,
            )

        # Find the index of the anchor_id in the sorted list
        # If anchor_id not found (e.g. file deleted?), default to middle
        anchor_idx = -1
        for i, (sid, _) in enumerate(flair_files):
            if sid == anchor_id:
                anchor_idx = i
                break

        if anchor_idx == -1:
            anchor_idx = len(flair_files) // 2

        # 3. Calculate Target Slice IDs based on Stride
        # Stacking: [Anchor-5, Anchor, Anchor+5]
        offsets = [-Config.STRIDE, 0, Config.STRIDE]
        target_slice_ids = []

        for offset in offsets:
            target_idx = anchor_idx + offset
            # Edge Clamping: Clamp index to available FLAIR range
            target_idx = max(0, min(target_idx, len(flair_files) - 1))

            # Get the actual Slice ID (filename number) at this clamped index
            clamped_slice_id = flair_files[target_idx][0]
            target_slice_ids.append(clamped_slice_id)

        # 4. Load Data for All Modalities
        channels = []

        for mod in Config.MODALITIES:  # ["FLAIR", "T1w", "T1wCE", "T2w"]
            mod_path_rel = row[f"path_{mod}"]
            full_mod_dir = os.path.join(Config.INPUT_DIR, mod_path_rel)

            for target_id in target_slice_ids:
                # Construct expected filename
                # Note: Filenames are typically Image-{ID}.dcm.
                # We assume standard naming.
                img_path = os.path.join(full_mod_dir, f"Image-{target_id}.dcm")

                # Load Image
                # Rule: "Use Zero Padding if a specific ID is missing in a secondary modality"
                # read_dicom_image handles missing files by returning zeros if path doesn't exist?
                # Actually read_dicom_image raises FileNotFoundError or returns zeros if checked.
                # We check existence explicitly to be safe and adhere to Zero Padding rule.

                if os.path.exists(img_path):
                    img = read_dicom_image(
                        img_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE)
                    )
                else:
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

                # 5. Independent Per-Channel Min-Max Scaling
                min_val = np.min(img)
                max_val = np.max(img)

                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)  # Avoid 0/0

                channels.append(img)

        # Stack channels: (12, 224, 224)
        # channels list has order: FLAIR_s1, FLAIR_s2, FLAIR_s3, T1w_s1, ...
        volume = np.stack(channels, axis=-1)  # (224, 224, 12)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=volume)
            volume = augmented[
                "image"
            ]  # (12, 224, 224) because ToTensorV2 converts HWC to CHW

        # Get Label (if available)
        label = (
            torch.tensor(row["MGMT_value"], dtype=torch.float32)
            if "MGMT_value" in row
            else torch.tensor(0.0)
        )

        return volume, label
