import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import library.config as C
import library.utils as U


def extract_slice_id(filename):
    """
    Extracts the integer slice ID from a standard BraTS filename (e.g., 'Image-123.dcm').
    Returns -1 if the pattern does not match.
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return -1


def get_sorted_files(dir_path):
    """
    Returns a list of DICOM filenames in a directory, sorted numerically by their slice ID.
    """
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    files.sort(key=extract_slice_id)
    return files


class MGMTDataset(Dataset):
    """
    Dataset class for MGMT Promoter Methylation prediction.
    Implements Fidelity-Aligned ROI selection, Explicit Contrast Injection, and Global Caching.
    """

    def __init__(
        self,
        metadata_df,
        transform=None,
        cache_path=None,
        load_cached_data=True,
        is_test=False,
        debug_limit=None,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing subject IDs and paths.
            transform (A.Compose): Albumentations transform pipeline.
            cache_path (str): Base path for saving/loading cached .npy files.
            load_cached_data (bool): Whether to attempt loading from cache.
            is_test (bool): If True, dummy labels are returned.
            debug_limit (int): Limit the dataset size for debugging purposes.
        """
        self.metadata_df = metadata_df
        if debug_limit:
            self.metadata_df = self.metadata_df.iloc[:debug_limit].copy()

        self.transform = transform
        self.is_test = is_test

        # Define cache file paths
        self.data_cache_file = None
        self.label_cache_file = None
        self.ids_cache_file = None

        if cache_path:
            # Ensure the directory exists
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            base_name = os.path.splitext(cache_path)[0]
            self.data_cache_file = f"{base_name}_data.npy"
            self.label_cache_file = f"{base_name}_labels.npy"
            self.ids_cache_file = f"{base_name}_ids.npy"

        self.data = None
        self.labels = None
        self.bra_ids = None

        # Caching Logic
        loaded = False
        if load_cached_data and self.data_cache_file:
            if os.path.exists(self.data_cache_file) and os.path.exists(
                self.label_cache_file
            ):
                try:
                    print(f"Loading cached data from {self.data_cache_file}...")
                    self.data = np.load(self.data_cache_file)
                    self.labels = np.load(self.label_cache_file)
                    if os.path.exists(self.ids_cache_file):
                        self.bra_ids = np.load(self.ids_cache_file)
                    loaded = True
                    print(f"Successfully loaded {len(self.data)} samples.")
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reprocessing...")

        if not loaded:
            print("Processing dataset from scratch...")
            self._process_dataset()

            # Save to cache if path is provided
            if self.data_cache_file:
                print(f"Saving cache to {self.data_cache_file}...")
                np.save(self.data_cache_file, self.data)
                np.save(self.label_cache_file, self.labels)
                np.save(self.ids_cache_file, self.bra_ids)

    def _process_dataset(self):
        """
        Iterates through the metadata, processes each subject, and aggregates the results.
        """
        data_list = []
        label_list = []
        id_list = []

        for idx, row in self.metadata_df.iterrows():
            bra_id = row["BraTS21ID"]
            # Get Label (use 0.5 as dummy for test set if column missing)
            label = row["MGMT_value"] if "MGMT_value" in row else 0.5

            # Construct absolute paths
            paths = {
                "FLAIR": os.path.join(C.INPUT_DIR, row["path_FLAIR"]),
                "T1w": os.path.join(C.INPUT_DIR, row["path_T1w"]),
                "T1wCE": os.path.join(C.INPUT_DIR, row["path_T1wCE"]),
            }

            try:
                volume = self._process_subject(paths)
            except Exception as e:
                print(f"Error processing subject {bra_id}: {e}")
                # Fallback: Return zero volume
                volume = np.zeros(
                    (C.IMG_SIZE, C.IMG_SIZE, C.INPUT_CHANNELS), dtype=np.float32
                )

            data_list.append(volume)
            label_list.append(label)
            id_list.append(bra_id)

        self.data = np.array(data_list, dtype=np.float32)
        self.labels = np.array(label_list, dtype=np.float32)
        self.bra_ids = np.array(id_list, dtype=np.int64)

    def _process_subject(self, paths):
        """
        Processes a single subject:
        1. Finds Anchor slice in FLAIR (Max Integral in 15-85% depth).
        2. Aligns T1w and T1wCE to Anchor.
        3. Computes Delta-T1 (T1wCE - T1w).
        4. Stacks and Normalizes.
        """
        # 1. Anchor Selection (FLAIR)
        flair_files = get_sorted_files(paths["FLAIR"])

        if not flair_files:
            return np.zeros(
                (C.IMG_SIZE, C.IMG_SIZE, C.INPUT_CHANNELS), dtype=np.float32
            )

        num_slices = len(flair_files)
        # Define ROI Depth Range
        start_idx = int(num_slices * C.ROI_DEPTH_RANGE[0])
        end_idx = int(num_slices * C.ROI_DEPTH_RANGE[1])

        # Safety check for small volumes
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_slices

        # Identify ROI files
        roi_files = flair_files[start_idx:end_idx]

        anchor_file_idx = -1

        if not roi_files:
            # Fallback to middle slice if ROI is empty
            anchor_file_idx = num_slices // 2
        else:
            # Find slice with maximum intensity sum
            best_sum = -1.0
            best_local_idx = 0

            for i, f in enumerate(roi_files):
                f_path = os.path.join(paths["FLAIR"], f)
                # Load raw image
                img = U.load_dicom_raw(f_path, C.IMG_SIZE)
                s = np.sum(img)
                if s > best_sum:
                    best_sum = s
                    best_local_idx = i

            anchor_file_idx = start_idx + best_local_idx

        # Get the integer ID of the anchor slice (e.g., 100 from Image-100.dcm)
        anchor_filename = flair_files[anchor_file_idx]
        anchor_id = extract_slice_id(anchor_filename)

        # 2. Define Target Slice IDs
        # Stride: [anchor-5, anchor, anchor+5]
        offsets = [-C.STRIDE, 0, C.STRIDE]
        target_ids = [anchor_id + off for off in offsets]

        # 3. Load Channels and Compute Delta
        # Helper to load specific ID
        def load_slice(modality, slice_id):
            fname = f"Image-{slice_id}.dcm"
            fpath = os.path.join(paths[modality], fname)
            return U.load_dicom_raw(fpath, C.IMG_SIZE)

        flair_imgs = [load_slice("FLAIR", tid) for tid in target_ids]
        t1w_imgs = [load_slice("T1w", tid) for tid in target_ids]
        t1wce_imgs = [load_slice("T1wCE", tid) for tid in target_ids]

        # Explicit Contrast Injection: Delta-T1 = T1wCE - T1w
        delta_imgs = []
        for t1, t1ce in zip(t1w_imgs, t1wce_imgs):
            d = t1ce - t1
            delta_imgs.append(d)

        # Stack Order: FLAIR (3), T1w (3), T1wCE (3), Delta (3) -> Total 12
        all_slices = flair_imgs + t1w_imgs + t1wce_imgs + delta_imgs

        # 4. Independent Per-Channel Normalization
        normalized_slices = []
        for img in all_slices:
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = np.zeros_like(img)
            normalized_slices.append(img)

        # Stack into (H, W, C) for Albumentations compatibility
        volume = np.stack(normalized_slices, axis=-1)
        return volume

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.data[idx]  # (H, W, 12)
        label = self.labels[idx]

        if self.transform:
            # Albumentations expects HWC
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W) via ToTensorV2
        else:
            # Manual conversion if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    transforms = []

    if phase == "train":
        transforms.append(A.HorizontalFlip(p=C.AUG_HFLIP_PROB))
        transforms.append(A.VerticalFlip(p=C.AUG_VFLIP_PROB))

        # Rotation with Reflection Padding
        border_mode = (
            cv2.BORDER_REFLECT if C.AUG_REFLECTION_PADDING else cv2.BORDER_CONSTANT
        )
        transforms.append(
            A.Rotate(
                limit=C.AUG_ROTATION_DEGREES, border_mode=border_mode, value=0, p=0.5
            )
        )

    # Always convert to Tensor at the end (HWC -> CHW)
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=C.PIN_MEMORY,
    )
