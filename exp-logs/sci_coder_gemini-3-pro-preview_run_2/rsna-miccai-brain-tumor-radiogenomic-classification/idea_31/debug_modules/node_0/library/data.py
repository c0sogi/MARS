import os
import re
import random
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import load_dicom_robust, resize_image, normalize_minmax


class MRIDataset(Dataset):
    """
    Dataset class for Glioblastoma Genetic Subtype Prediction.
    Implements Raw-Integral ROI selection, Volumetric Anchor Jitter, and
    robust multi-modal stacking.
    """

    def __init__(self, metadata_df, phase="train", load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing dataset metadata.
            phase (str): 'train', 'val', or 'test'. Controls augmentation and return values.
            load_cached_data (bool): Whether to load/save ROI anchors from cache.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.phase = phase
        self.load_cached_data = load_cached_data

        # Define Augmentations
        # We use Albumentations. It supports multi-channel images if shape is (H, W, C).
        if self.phase == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(
                        limit=Config.ROTATION_DEGREES,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                ]
            )
        else:
            self.transform = None

        # Precompute or Load ROI Anchors
        self.roi_anchors = self._precompute_anchors()

    def _get_file_list(self, dir_path):
        """Helper to get sorted list of DICOM files from a directory."""
        if not os.path.exists(dir_path):
            return []
        files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
        # Sort numerically by extracting the number from 'Image-XXX.dcm'
        files.sort(
            key=lambda x: (
                int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 0
            )
        )
        return files

    def _precompute_anchors(self):
        """
        Determines the optimal anchor slice index for each subject based on
        FLAIR modality intensity sum. Caches results to Parquet.
        """
        cache_file = os.path.join(Config.CACHE_DIR, "roi_cache.parquet")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_file):
            try:
                cache_df = pd.read_parquet(cache_file)
                # Convert to dict for fast lookup: {BraTS21ID: anchor_index}
                return dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(
            f"Computing ROI anchors for {len(self.df)} subjects (Phase: {self.phase})..."
        )
        anchors = {}

        for idx, row in self.df.iterrows():
            subject_id = row["BraTS21ID"]

            # Construct full path to FLAIR directory
            # Metadata paths are relative to INPUT_DIR
            flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
            files = self._get_file_list(flair_path)
            num_files = len(files)

            if num_files == 0:
                anchors[subject_id] = 0
                continue

            # Define search range (15% - 85%)
            start_idx = int(num_files * Config.ROI_DEPTH_RANGE[0])
            end_idx = int(num_files * Config.ROI_DEPTH_RANGE[1])

            # Ensure valid range
            start_idx = max(0, start_idx)
            end_idx = min(num_files, end_idx)
            if start_idx >= end_idx:
                start_idx = 0
                end_idx = num_files

            max_intensity_sum = -1.0
            best_idx = start_idx  # Default to start of range

            # Iterate through the range to find the slice with max intensity sum
            # Note: We use the robust loader but only for the ROI modality (FLAIR)
            for i in range(start_idx, end_idx):
                f_path = os.path.join(flair_path, files[i])
                # We use the robust loader here to be safe
                img = load_dicom_robust(f_path)
                current_sum = np.sum(img)

                if current_sum > max_intensity_sum:
                    max_intensity_sum = current_sum
                    best_idx = i

            anchors[subject_id] = best_idx

        # 3. Save to cache (merge with existing if possible, but here we just overwrite/save what we found)
        # To make it robust across multiple runs (e.g. train vs test), we should ideally append,
        # but for simplicity in this task, we save what we computed.
        # If the file exists, we load it first to update it.
        final_anchors = anchors.copy()
        if os.path.exists(cache_file):
            try:
                existing_df = pd.read_parquet(cache_file)
                existing_dict = dict(
                    zip(existing_df["BraTS21ID"], existing_df["anchor_index"])
                )
                existing_dict.update(anchors)
                final_anchors = existing_dict
            except:
                pass

        # Save
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        out_df = pd.DataFrame(
            list(final_anchors.items()), columns=["BraTS21ID", "anchor_index"]
        )
        out_df.to_parquet(cache_file, index=False)

        return final_anchors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # 1. Determine Anchor
        anchor = self.roi_anchors.get(subject_id, 0)

        # 2. Apply Volumetric Anchor Jitter (Training Only)
        if self.phase == "train" and random.random() < Config.ANCHOR_JITTER_PROB:
            jitter = random.choice(Config.ANCHOR_JITTER_RANGE)
            anchor += jitter

        # 3. Define Slice Indices
        # [Anchor - Stride, Anchor, Anchor + Stride]
        stride = Config.STRIDE
        slice_indices = [anchor - stride, anchor, anchor + stride]

        # 4. Load Data
        channels = []

        for mod in Config.MODALITIES:  # FLAIR, T1w, T1wCE, T2w
            mod_dir = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            files = self._get_file_list(mod_dir)
            num_files = len(files)

            for slice_idx in slice_indices:
                # Boundary Clamping (Edge Clamping)
                # If index is out of bounds, clamp to nearest valid index
                read_idx = max(0, min(slice_idx, num_files - 1))

                if num_files > 0:
                    f_path = os.path.join(mod_dir, files[read_idx])
                    img = load_dicom_robust(f_path)
                else:
                    # Handle missing directory/files
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

                # Resize
                img = resize_image(img, size=Config.IMG_SIZE)

                # Normalize (Independent Per-Channel Min-Max)
                img = normalize_minmax(img)

                channels.append(img)

        # Stack channels: (12, H, W)
        # Order: FLAIR(3), T1w(3), T1wCE(3), T2w(3)
        image_tensor = np.stack(channels, axis=0)

        # 5. Augmentation
        if self.transform:
            # Albumentations expects (H, W, C)
            # Transpose: (C, H, W) -> (H, W, C)
            image_np = np.transpose(image_tensor, (1, 2, 0))

            augmented = self.transform(image=image_np)
            image_np = augmented["image"]

            # Transpose back: (H, W, C) -> (C, H, W)
            image_tensor = np.transpose(image_np, (2, 0, 1))

        # Convert to torch tensor
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)

        # 6. Return
        if self.phase == "test":
            # For test set, return ID for submission mapping
            return image_tensor, subject_id
        else:
            # For train/val, return label
            label = row["MGMT_value"]
            return image_tensor, torch.tensor(label, dtype=torch.float32)


def get_dataloader(metadata_df, phase, batch_size=None, shuffle=None, num_workers=None):
    """
    Factory function to create a DataLoader.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if shuffle is None:
        shuffle = phase == "train"

    dataset = MRIDataset(metadata_df, phase=phase, load_cached_data=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
