import os
import re
import random
import numpy as np
import cv2
import torch
import pandas as pd
from library import config

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_slice(path, size=256):
    """
    Loads a DICOM slice, resizes it, and normalizes it to [0, 1].
    Attempts to use OpenCV first, then pydicom.
    """
    img = None

    # 1. Try OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # 2. Try pydicom if OpenCV failed
    if img is None and HAS_PYDICOM:
        try:
            dicom = pydicom.dcmread(path)
            img = dicom.pixel_array
        except Exception:
            img = None

    # 3. Fallback: Return zeros if read fails
    if img is None:
        if isinstance(size, int):
            return np.zeros((size, size), dtype=np.float32)
        else:
            return np.zeros(size, dtype=np.float32)

    # Handle multi-channel (convert to grayscale if needed)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    # Normalize to [0, 1]
    img = img.astype(np.float32)
    denom = img.max() - img.min()
    if denom > 0:
        img = (img - img.min()) / denom
    else:
        img = img - img.min()  # Results in zeros

    # Resize
    if isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = size

    img = cv2.resize(img, target_size)

    return img


def get_sorted_files(dir_path):
    """
    Returns sorted list of files in a directory based on Image-N.dcm numbering.
    """
    if not os.path.exists(dir_path):
        return []

    files = os.listdir(dir_path)
    # Filter for .dcm files
    files = [f for f in files if f.endswith(".dcm")]

    def extract_number(f):
        # Matches Image-123.dcm
        match = re.search(r"Image-(\d+)\.dcm", f)
        if match:
            return int(match.group(1))
        return 0

    return sorted(files, key=extract_number)


def compute_best_slices(df, cache_name="train", load_cached_data=True):
    """
    Computes the index of the FLAIR slice with maximum intensity for each subject.
    Caches the result to disk to speed up future runs.
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_name}_processed.parquet")

    # 1. Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached metadata from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Computing best slice indices for {cache_name} (this may take a while)...")

    best_indices = []
    num_slices_list = []

    for idx, row in df.iterrows():
        flair_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])

        files = get_sorted_files(flair_path)
        num_slices = len(files)

        if num_slices == 0:
            best_indices.append(0)
            num_slices_list.append(0)
            continue

        # Find max intensity slice
        max_intensity = -1
        best_idx = 0

        # Iterate through slices to find the brightest one (tumor/brain center)
        for i, f in enumerate(files):
            p = os.path.join(flair_path, f)
            try:
                # Use raw read for speed, skip full normalization
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is None and HAS_PYDICOM:
                    img = pydicom.dcmread(p).pixel_array

                if img is not None:
                    intensity = np.mean(img)
                    if intensity > max_intensity:
                        max_intensity = intensity
                        best_idx = i
            except Exception:
                continue

        best_indices.append(best_idx)
        num_slices_list.append(num_slices)

    df_processed = df.copy()
    df_processed["best_flair_index"] = best_indices
    df_processed["num_flair_slices"] = num_slices_list

    # 2. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_processed.to_parquet(cache_path)
    print(f"Saved processed metadata to {cache_path}")

    return df_processed


class MgmtDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, phase="train"):
        self.df = df
        self.transform = transform
        self.phase = phase
        self.modalities = config.MODALITIES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Determine Indices
        # ROI View: [best-1, best, best+1]
        best_idx = row.get("best_flair_index", 0)
        num_slices = row.get("num_flair_slices", 10)

        if num_slices < 1:
            num_slices = 1

        # ROI indices (clamped)
        roi_indices = [best_idx - 1, best_idx, best_idx + 1]
        roi_indices = [max(0, min(i, num_slices - 1)) for i in roi_indices]

        # Geometric View: 25%, 50%, 75%
        geo_indices = [
            int(num_slices * 0.25),
            int(num_slices * 0.50),
            int(num_slices * 0.75),
        ]
        geo_indices = [max(0, min(i, num_slices - 1)) for i in geo_indices]

        # 2. Load Images
        roi_channels = []
        geo_channels = []

        for mod in self.modalities:
            dir_path = os.path.join(config.INPUT_DIR, row[f"path_{mod}"])
            files = get_sorted_files(dir_path)
            mod_slices = len(files)

            if mod_slices == 0:
                # Missing modality? Return zeros
                zeros = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
                for _ in range(3):
                    roi_channels.append(zeros)
                for _ in range(3):
                    geo_channels.append(zeros)
                continue

            # Helper to load by index (handling count mismatch between modalities)
            def get_slice(target_idx_flair_space):
                # Map flair index to this modality
                if num_slices > 0:
                    ratio = target_idx_flair_space / num_slices
                    idx_mod = int(ratio * mod_slices)
                else:
                    idx_mod = 0

                idx_mod = max(0, min(idx_mod, mod_slices - 1))
                f_name = files[idx_mod]
                return load_dicom_slice(
                    os.path.join(dir_path, f_name), size=config.IMG_SIZE
                )

            # Load ROI
            for i in roi_indices:
                roi_channels.append(get_slice(i))

            # Load Geo
            for i in geo_indices:
                geo_channels.append(get_slice(i))

        # Stack: (C, H, W) -> C = 12 (3 slices * 4 mods)
        roi_tensor = np.stack(roi_channels, axis=0)
        geo_tensor = np.stack(geo_channels, axis=0)

        # 3. Augmentations
        if self.transform:
            # Transpose to HWC for albumentations
            roi_hwc = np.transpose(roi_tensor, (1, 2, 0))
            geo_hwc = np.transpose(geo_tensor, (1, 2, 0))

            # Apply transform independently to learn robust features
            res_roi = self.transform(image=roi_hwc)
            roi_hwc = res_roi["image"]

            res_geo = self.transform(image=geo_hwc)
            geo_hwc = res_geo["image"]

            # Back to CHW
            roi_tensor = np.transpose(roi_hwc, (2, 0, 1))
            geo_tensor = np.transpose(geo_hwc, (2, 0, 1))

        # Convert to torch
        roi_tensor = torch.from_numpy(roi_tensor).float()
        geo_tensor = torch.from_numpy(geo_tensor).float()

        # Get Label
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return roi_tensor, geo_tensor, target
        else:
            return roi_tensor, geo_tensor
