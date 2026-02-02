import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config, utils

# -----------------------------------------------------------------------------
# Helper Functions: IO & Processing
# -----------------------------------------------------------------------------


def read_dicom_fast(filepath):
    """
    Reads a DICOM file using raw binary tail-read to bypass header parsing issues.
    Assumes uint16 data at the end of the file.
    Resizes to config.IMG_SIZE using Area Interpolation.
    Returns float32 numpy array.
    """
    try:
        file_size = os.path.getsize(filepath)

        # Heuristic to determine resolution based on file size
        # 512x512x2 = 524,288 bytes. Files ~525KB are 512x512.
        # 256x256x2 = 131,072 bytes. Files ~132KB are 256x256.
        # Threshold set at 300KB to separate the two clusters.
        if file_size > 300000:
            shape = (512, 512)
        else:
            shape = (256, 256)

        num_bytes = shape[0] * shape[1] * 2

        with open(filepath, "rb") as f:
            f.seek(-num_bytes, 2)  # Seek from end of file
            buffer = f.read()

        if len(buffer) < num_bytes:
            # Fallback if file is truncated
            return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

        img = np.frombuffer(buffer, dtype=np.uint16).reshape(shape)

        # Resize to target dimension
        if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
            img = cv2.resize(
                img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )

        return img.astype(np.float32)

    except Exception:
        # Return black image on failure
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)


def parse_slice_id(filename):
    """Extracts the integer slice ID from a filename like 'Image-123.dcm'."""
    try:
        base = os.path.basename(filename)
        name_part = os.path.splitext(base)[0]  # e.g., Image-123
        num = int(name_part.split("-")[-1])
        return num
    except:
        return -1


# -----------------------------------------------------------------------------
# ROI Selection Logic (Consensus)
# -----------------------------------------------------------------------------


def compute_roi_consensus(df, input_dir):
    """
    Computes the anchor slice ID for each subject using Aligned-Consensus.
    Scans FLAIR and T1wCE, computes intensity sums, combines them,
    and picks the peak in the central 15-85% volume.
    """
    results = []

    for idx, row in df.iterrows():
        subject_id = row["BraTS21ID"]

        # Paths
        path_flair = os.path.join(input_dir, row["path_FLAIR"])
        path_t1wce = os.path.join(input_dir, row["path_T1wCE"])

        # Helper to map slice_id -> filepath
        def get_slice_map(folder_path):
            if not os.path.exists(folder_path):
                return {}
            files = os.listdir(folder_path)
            slice_map = {}
            for f in files:
                sid = parse_slice_id(f)
                if sid != -1:
                    slice_map[sid] = os.path.join(folder_path, f)
            return slice_map

        map_flair = get_slice_map(path_flair)
        map_t1wce = get_slice_map(path_t1wce)

        # Find intersection of explicit slice IDs to ensure alignment
        common_ids = sorted(list(set(map_flair.keys()) & set(map_t1wce.keys())))

        if not common_ids:
            # Fallback: use middle of FLAIR if no intersection
            if map_flair:
                ids = sorted(list(map_flair.keys()))
                anchor = ids[len(ids) // 2]
            else:
                anchor = 0
            results.append({"BraTS21ID": subject_id, "anchor_slice_id": anchor})
            continue

        # Calculate intensity profiles
        intensities_flair = []
        intensities_t1wce = []

        for sid in common_ids:
            img_f = read_dicom_fast(map_flair[sid])
            img_t = read_dicom_fast(map_t1wce[sid])
            intensities_flair.append(np.sum(img_f))
            intensities_t1wce.append(np.sum(img_t))

        # Normalize profiles to [0, 1]
        def normalize_profile(arr):
            arr = np.array(arr)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 0:
                return (arr - mn) / (mx - mn)
            return np.zeros_like(arr)

        prof_f = normalize_profile(intensities_flair)
        prof_t = normalize_profile(intensities_t1wce)

        # Consensus
        consensus = prof_f + prof_t

        # Restrict to 15%-85% depth
        n_slices = len(common_ids)
        start_idx = int(n_slices * config.ROI_MIN_DEPTH)
        end_idx = int(n_slices * config.ROI_MAX_DEPTH)

        if start_idx >= end_idx:
            start_idx = 0
            end_idx = n_slices

        valid_consensus = consensus[start_idx:end_idx]

        if len(valid_consensus) > 0:
            peak_offset = np.argmax(valid_consensus)
            peak_idx = start_idx + peak_offset
            anchor = common_ids[peak_idx]
        else:
            anchor = common_ids[n_slices // 2]

        results.append({"BraTS21ID": subject_id, "anchor_slice_id": anchor})

    return pd.DataFrame(results)


def get_roi_cache_data(metadata_df, cache_file, load_cached_data=True):
    """
    Manages the ROI cache. Loads if available and requested, else computes and saves.
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Try loading
    if load_cached_data and os.path.exists(cache_file):
        try:
            cache_df = pd.read_parquet(cache_file)
            # Verify coverage
            required_ids = set(metadata_df["BraTS21ID"])
            cached_ids = set(cache_df["BraTS21ID"])
            if required_ids.issubset(cached_ids):
                return cache_df
        except Exception:
            pass  # Fall through to recompute

    # Compute
    print("Computing ROI Cache (Consensus Strategy)...")
    new_cache_df = compute_roi_consensus(metadata_df, config.INPUT_DIR)

    # Merge with existing if possible to preserve other entries
    if os.path.exists(cache_file):
        try:
            existing_df = pd.read_parquet(cache_file)
            # Remove entries that are in the new batch to update them, keep others
            existing_df = existing_df[
                ~existing_df["BraTS21ID"].isin(new_cache_df["BraTS21ID"])
            ]
            final_df = pd.concat([existing_df, new_cache_df], ignore_index=True)
        except:
            final_df = new_cache_df
    else:
        final_df = new_cache_df

    final_df.to_parquet(cache_file, index=False)
    return final_df


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class BraTSDataset(Dataset):
    def __init__(self, metadata_df, roi_cache_df, phase="train", transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.roi_map = dict(
            zip(roi_cache_df["BraTS21ID"], roi_cache_df["anchor_slice_id"])
        )
        self.phase = phase
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        subject_id = row["BraTS21ID"]

        # 1. Get Anchor Slice
        anchor = self.roi_map.get(subject_id, 0)

        # 2. Define Target IDs (Logical-Geometric Stacking)
        stride = config.STRIDE
        target_ids = [anchor - stride, anchor, anchor + stride]

        channels = []

        # 3. Load Data for each Modality
        for mod in self.modalities:
            path_col = f"path_{mod}"
            mod_dir = os.path.join(config.INPUT_DIR, row[path_col])

            # Map IDs to files
            available_files = {}
            if os.path.exists(mod_dir):
                for f in os.listdir(mod_dir):
                    sid = parse_slice_id(f)
                    if sid != -1:
                        available_files[sid] = os.path.join(mod_dir, f)

            # Retrieve slices
            for tid in target_ids:
                fpath = available_files.get(tid)

                # Edge Clamping / Nearest Neighbor Fallback
                if fpath is None and available_files:
                    sids = sorted(available_files.keys())
                    closest = min(sids, key=lambda x: abs(x - tid))
                    fpath = available_files[closest]

                if fpath:
                    img = read_dicom_fast(fpath)
                else:
                    img = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

                # Independent Per-Channel Min-Max Normalization
                mn, mx = img.min(), img.max()
                if mx - mn > 0:
                    img = (img - mn) / (mx - mn + 1e-6)
                else:
                    img = img - mn  # Zero

                channels.append(img)

        # Stack -> (H, W, 12)
        img_stack = np.stack(channels, axis=-1)

        # 4. Augmentation
        if self.phase == "train" and self.transform:
            augmented = self.transform(image=img_stack)
            img_stack = augmented["image"]

        # 5. Convert to Tensor (C, H, W)
        if not isinstance(img_stack, torch.Tensor):
            img_stack = torch.from_numpy(img_stack).permute(2, 0, 1).float()

        # 6. Return
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_stack, label
        else:
            # Test set
            return img_stack, torch.tensor(-1.0)


# -----------------------------------------------------------------------------
# Data Loader Factory
# -----------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True):
    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Combine to ensure ROI cache covers everything
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Prepare ROI Cache
    cache_path = os.path.join(config.CACHE_DIR, "roi_cache.parquet")
    roi_cache = get_roi_cache_data(
        df_all, cache_path, load_cached_data=load_cached_data
    )

    # Define Transforms
    # Albumentations handles multi-channel images (H, W, C) automatically for geometric transforms
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Cite solution_lesson_node_00091: Prefer Reflection Padding over Zero-Padding for Geometric Augmentations
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # Create Datasets
    train_dataset = BraTSDataset(
        df_train, roi_cache, phase="train", transform=train_transform
    )
    val_dataset = BraTSDataset(df_val, roi_cache, phase="val", transform=val_transform)
    test_dataset = BraTSDataset(
        df_test, roi_cache, phase="test", transform=val_transform
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
