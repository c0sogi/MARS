import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def read_dicom_robust(filepath):
    """
    Reads a DICOM file using Raw Binary Tail-Read strategy.
    Bypasses header parsing by directly reading pixel bytes from the end of the file.
    Assumes uncompressed uint16 data.
    """
    try:
        file_size = os.path.getsize(filepath)

        # Define expected raw sizes (H * W * 2 bytes)
        size_512 = 512 * 512 * 2
        size_256 = 256 * 256 * 2

        img = None

        # Attempt to read as 512x512
        if file_size >= size_512:
            offset = file_size - size_512
            with open(filepath, "rb") as f:
                f.seek(offset)
                data = f.read(size_512)
            arr = np.frombuffer(data, dtype=np.uint16)
            if arr.size == 512 * 512:
                img = arr.reshape(512, 512)

        # Attempt to read as 256x256 if 512 failed or size suggests smaller
        if img is None and file_size >= size_256:
            offset = file_size - size_256
            with open(filepath, "rb") as f:
                f.seek(offset)
                data = f.read(size_256)
            arr = np.frombuffer(data, dtype=np.uint16)
            if arr.size == 256 * 256:
                img = arr.reshape(256, 256)

        # Fallback: Return black image if reading failed
        if img is None:
            return np.zeros(Config.IMG_SIZE, dtype=np.float32)

        # Convert to float32 for processing
        img = img.astype(np.float32)

        # Resize using Area Interpolation (Low-pass filter)
        if img.shape != Config.IMG_SIZE:
            img = cv2.resize(img, Config.IMG_SIZE, interpolation=cv2.INTER_AREA)

        return img

    except Exception:
        # Return zero array on any IO error
        return np.zeros(Config.IMG_SIZE, dtype=np.float32)


def get_sorted_image_files(directory):
    """
    Returns a sorted list of (image_number, filename) tuples from a directory.
    Sorts based on the numeric value in 'Image-X.dcm'.
    """
    try:
        files = os.listdir(directory)
    except OSError:
        return []

    files = [f for f in files if f.endswith(".dcm")]

    files_with_idx = []
    for f in files:
        # Extract number from Image-X.dcm
        match = re.search(r"Image-(\d+)", f)
        if match:
            idx = int(match.group(1))
            files_with_idx.append((idx, f))

    # Sort numerically
    files_with_idx.sort(key=lambda x: x[0])
    return files_with_idx


def generate_roi_cache(metadata_dfs, load_cached_data=True):
    """
    Generates or loads a cache of the best ROI anchor index for each subject.
    Strategy: Raw-Integral-Reference (Max Sum of Raw FLAIR Intensity).
    """
    cache_path = os.path.join(Config.WORKING_DIR, "roi_cache.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            roi_map = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))
            print(f"Loaded ROI cache from {cache_path} with {len(roi_map)} entries.")
            return roi_map
        except Exception as e:
            print(f"Cache load failed ({e}). Regenerating...")

    # 2. Regenerate cache
    print("Generating ROI cache using Raw-Integral-Reference strategy...")
    roi_map = {}

    # Consolidate unique subjects
    unique_subjects = {}
    for df in metadata_dfs:
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                unique_subjects[row["BraTS21ID"]] = row

    for sid, row in unique_subjects.items():
        flair_dir = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])

        files = get_sorted_image_files(flair_dir)

        if not files:
            roi_map[sid] = 0
            continue

        num_files = len(files)
        # Depth Search Bounds
        start_idx = int(num_files * Config.ANCHOR_MIN_QUANTILE)
        end_idx = int(num_files * Config.ANCHOR_MAX_QUANTILE)

        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_files

        best_sum = -1.0
        # Default to middle slice if search fails
        best_file_num = files[num_files // 2][0]

        # Iterate candidates
        for i in range(start_idx, end_idx):
            file_num, fname = files[i]
            full_path = os.path.join(flair_dir, fname)

            # Read raw
            img = read_dicom_robust(full_path)

            # Integral Metric (Sum of Intensity)
            current_sum = np.sum(img)

            if current_sum > best_sum:
                best_sum = current_sum
                best_file_num = file_num

        roi_map[sid] = best_file_num

    # 3. Save cache
    try:
        cache_df = pd.DataFrame(
            list(roi_map.items()), columns=["BraTS21ID", "anchor_index"]
        )
        cache_df.to_parquet(cache_path)
        print(f"Saved ROI cache to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save ROI cache: {e}")

    return roi_map


class MGMTDataset(Dataset):
    def __init__(self, df, roi_map, transform=None):
        self.df = df
        self.roi_map = roi_map
        self.transform = transform
        self.modalities = Config.MODALITIES
        self.stride = Config.STRIDE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        # Retrieve anchor
        anchor = self.roi_map.get(sid, -1)

        # Define slice indices: [Previous, Anchor, Next]
        # Using absolute file indices (Image-X)
        file_indices = [anchor - self.stride, anchor, anchor + self.stride]

        channels = []

        # Iterate Modalities -> Slices to form groups
        # Order: Mod1_S1, Mod1_S2, Mod1_S3, Mod2_S1, ...
        for mod in self.modalities:
            mod_dir = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])

            for f_idx in file_indices:
                img = None

                # Check if file index is valid (Image numbers are >= 1)
                if f_idx >= 1:
                    fpath = os.path.join(mod_dir, f"Image-{f_idx}.dcm")
                    if os.path.exists(fpath):
                        img = read_dicom_robust(fpath)
                    else:
                        # Edge Clamping: If neighbor missing, try anchor
                        if f_idx != anchor:
                            anchor_path = os.path.join(mod_dir, f"Image-{anchor}.dcm")
                            if os.path.exists(anchor_path):
                                img = read_dicom_robust(anchor_path)

                # If still None (e.g., negative index or anchor missing), use zero padding
                if img is None:
                    img = np.zeros(Config.IMG_SIZE, dtype=np.float32)

                channels.append(img)

        # Stack to (H, W, C)
        volume = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=volume)
            volume = augmented["image"]

        # Ensure Tensor (C, H, W)
        if not isinstance(volume, torch.Tensor):
            volume = ToTensorV2()(image=volume)["image"]

        # Conservative Min-Max Scaling [0, 1]
        v_min = volume.min()
        v_max = volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = volume - v_min  # Zero out

        # Get Label
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)

        return volume, label


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns the training, validation, and test DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Prepare ROI Cache
    roi_map = generate_roi_cache(
        [train_df, val_df, test_df], load_cached_data=load_cached_data
    )

    # Define Transforms
    # Train: Flip + Rotate
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            ToTensorV2(),
        ]
    )

    # Val/Test: Just Tensor conversion
    val_transform = A.Compose([ToTensorV2()])

    # Instantiate Datasets
    train_dataset = MGMTDataset(train_df, roi_map, transform=train_transform)
    val_dataset = MGMTDataset(val_df, roi_map, transform=val_transform)
    test_dataset = MGMTDataset(test_df, roi_map, transform=val_transform)

    # Instantiate DataLoaders
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
