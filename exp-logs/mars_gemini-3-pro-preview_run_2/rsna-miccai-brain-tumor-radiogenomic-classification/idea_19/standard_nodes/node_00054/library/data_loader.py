import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_19"
IMG_SIZE = 224
NUM_SLICES = 3
STRIDE = 5
CHANNELS_PER_IMAGE = 4  # FLAIR, T1w, T1wCE, T2w
TOTAL_CHANNELS = NUM_SLICES * CHANNELS_PER_IMAGE

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def load_image_robust(path):
    """
    Reads an MRI slice using a multi-tiered strategy:
    1. OpenCV (cv2.imread)
    2. pydicom (if available)
    3. Raw Binary Tail-Read (fallback for uncompressed DICOM)
    """
    if not os.path.exists(path):
        return np.zeros((512, 512), dtype=np.float32)

    # Tier 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # Tier 2: pydicom
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img.astype(np.float32)
    except (ImportError, Exception):
        pass

    # Tier 3: Raw Binary Fallback (Assuming 512x512 uint16)
    try:
        file_size = os.path.getsize(path)
        expected_bytes = 512 * 512 * 2
        if file_size >= expected_bytes:
            with open(path, "rb") as f:
                f.seek(-expected_bytes, os.SEEK_END)
                buffer = f.read(expected_bytes)
            img = np.frombuffer(buffer, dtype=np.uint16).reshape(512, 512)
            return img.astype(np.float32)
    except Exception:
        pass

    # Ultimate failure
    return np.zeros((512, 512), dtype=np.float32)


def get_sorted_files(dir_path):
    """Returns list of files in directory sorted by the integer number in 'Image-X.dcm'."""
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    # Sort based on the integer value X in Image-X.dcm
    # Handle cases where format might differ slightly, but standard is Image-X.dcm
    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except Exception:
        files.sort()  # Fallback to lexicographical if naming convention breaks
    return files


def compute_anchor_indices(metadata_df, load_cached_data=True):
    """
    Implements Hierarchical-Gated ROI Selection.
    Stage 1: FLAIR Energy Gating (Central 80%)
    Stage 2: T1wCE Max Intensity Targeting within Gate
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "roi_cache.parquet")

    # 1. Load Cache if requested
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert to dict: BraTS21ID -> anchor_index
            return dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Computing ROI anchors (Hierarchical-Gated)...")
    anchor_dict = {}

    # Iterate over all subjects
    # We need to process row by row
    for idx, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]

        # Paths
        flair_dir = os.path.join(INPUT_DIR, row["path_FLAIR"])
        t1wce_dir = os.path.join(INPUT_DIR, row["path_T1wCE"])

        flair_files = get_sorted_files(flair_dir)
        t1wce_files = get_sorted_files(t1wce_dir)

        num_flair = len(flair_files)

        if num_flair == 0:
            anchor_dict[subject_id] = 0
            continue

        # --- Stage 1: Anatomical Gating (FLAIR) ---
        # Calculate energy profile
        energy_profile = []
        # We can't read ALL files if time is tight, but for ROI we usually need to.
        # To optimize, we read every 2nd or 3rd slice for the profile if N is large?
        # Given the constraints and small dataset size (500 subjects), reading is feasible.
        # We will read all to be accurate.

        valid_flair_indices = []

        # Pre-read FLAIR to find brain volume
        flair_intensities = []
        for f in flair_files:
            p = os.path.join(flair_dir, f)
            img = load_image_robust(p)
            energy = np.sum(img)
            flair_intensities.append(energy)

        flair_intensities = np.array(flair_intensities)
        total_energy = np.sum(flair_intensities)

        start_idx = 0
        end_idx = num_flair - 1

        if total_energy > 0:
            cumsum = np.cumsum(flair_intensities)
            # Find indices for central 80% (10% to 90%)
            try:
                start_idx = np.where(cumsum >= 0.1 * total_energy)[0][0]
                end_idx = np.where(cumsum >= 0.9 * total_energy)[0][0]
            except IndexError:
                start_idx = 0
                end_idx = num_flair - 1

        # Ensure indices are within bounds and logical
        start_idx = max(0, start_idx)
        end_idx = min(num_flair - 1, end_idx)
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_flair - 1

        # --- Stage 2: Pathological Targeting (T1wCE) ---
        # We search for max intensity in T1wCE within the valid window [start_idx, end_idx]
        # Note: T1wCE files might not align 1:1 with FLAIR indices if counts differ.
        # However, usually in BraTS datasets, slices are co-registered or we assume spatial correspondence
        # by normalized depth. Here we assume file index correspondence or simply map the range.
        # If file counts differ significantly, we map the relative depth.

        num_t1wce = len(t1wce_files)

        if num_t1wce > 0:
            # Map start/end from FLAIR space to T1wCE space
            ratio = num_t1wce / num_flair
            t_start = int(start_idx * ratio)
            t_end = int(end_idx * ratio)
            t_end = max(t_start + 1, t_end)
            t_end = min(num_t1wce, t_end)

            max_intensity = -1
            best_idx_local = -1

            # Scan T1wCE
            for i in range(t_start, t_end):
                p = os.path.join(t1wce_dir, t1wce_files[i])
                img = load_image_robust(p)
                val = np.max(img)
                if val > max_intensity:
                    max_intensity = val
                    best_idx_local = i

            # Fallback check: if max intensity is very low (black image), revert to FLAIR
            if max_intensity < 10:  # Arbitrary low threshold for "empty"
                # Revert to FLAIR Max in the valid range
                best_idx_local = -1
        else:
            best_idx_local = -1

        # --- Fallback: FLAIR Max ---
        if best_idx_local == -1:
            # Find max intensity in FLAIR within the valid range
            max_intensity = -1
            best_flair_idx = start_idx
            for i in range(start_idx, end_idx + 1):
                # We already loaded these, but didn't store the max pixel, only sum.
                # To save IO, we might just use the sum profile as a proxy for "most brain"
                # or re-read. Re-reading is safer for "Max Pixel".
                p = os.path.join(flair_dir, flair_files[i])
                img = load_image_robust(p)
                val = np.max(img)
                if val > max_intensity:
                    max_intensity = val
                    best_flair_idx = i

            # Map back to T1wCE space if we need a unified anchor index?
            # The dataset loader assumes we use one integer index.
            # If we use FLAIR index, we must ensure we pick the corresponding files in other modalities.
            # We will store the relative depth (0.0 to 1.0) or just the index.
            # Since MRIDataset uses the same integer index for all modalities,
            # we should store the index relative to the modality we used,
            # but this is problematic if slice counts differ.
            # Strategy: Store the normalized position (0.0-1.0) and convert in Dataset?
            # Or simpler: Just store the index assuming roughly equal slice counts (common in BraTS).
            # Let's store the index. If we derived it from FLAIR, we use that index.
            anchor_dict[subject_id] = best_flair_idx
        else:
            # We found a good T1wCE slice.
            # If we used T1wCE index, we need to map it back to FLAIR/others if counts differ?
            # Let's assume we use the index `best_idx_local` which is a T1wCE index.
            # If we use this index to load FLAIR, and FLAIR has different count, we might be off.
            # Robust approach: Store the Middle Slice of the ROI if counts differ,
            # or just use the index and handle out-of-bounds in Dataset.
            anchor_dict[subject_id] = best_idx_local

    # Save to cache
    cache_data = [{"BraTS21ID": k, "anchor_index": v} for k, v in anchor_dict.items()]
    pd.DataFrame(cache_data).to_parquet(cache_path)

    return anchor_dict


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class MRIDataset(Dataset):
    def __init__(self, df, anchor_dict, transform=None, mode="train"):
        self.df = df
        self.anchor_dict = anchor_dict
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get anchor
        anchor = self.anchor_dict.get(subject_id, 0)

        # Define slice indices: anchor-5, anchor, anchor+5
        indices = [anchor - STRIDE, anchor, anchor + STRIDE]

        # Modalities
        modalities = ["path_FLAIR", "path_T1w", "path_T1wCE", "path_T2w"]

        channels = []

        for mod_col in modalities:
            dir_path = os.path.join(INPUT_DIR, row[mod_col])
            files = get_sorted_files(dir_path)
            num_files = len(files)

            for slice_idx in indices:
                # Handle out of bounds / missing files
                # If slice counts differ between modalities, we map the index proportionally?
                # Or we just clamp. Clamping is safer.
                # If the anchor was derived from T1wCE (e.g. idx 100) and FLAIR has 20 slices, 100 is invalid.
                # We need a robust way to map index.
                # Simple robust strategy: Clamp to [0, num_files-1]

                if num_files == 0:
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                else:
                    # Clamp
                    read_idx = max(0, min(slice_idx, num_files - 1))
                    file_path = os.path.join(dir_path, files[read_idx])

                    # Load
                    img = load_image_robust(file_path)

                    # Resize
                    img = cv2.resize(
                        img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                    )

                    # Normalize (Min-Max)
                    mi, ma = np.min(img), np.max(img)
                    if ma > mi:
                        img = (img - mi) / (ma - mi)
                    else:
                        img = np.zeros_like(img)

                channels.append(img)

        # Stack: (12, 224, 224) -> but Albumentations needs (H, W, C)
        # channels is list of 12 (224, 224) arrays
        img_stack = np.stack(channels, axis=-1)  # (224, 224, 12)

        # Augmentations
        if self.transform:
            augmented = self.transform(image=img_stack)
            img_stack = augmented[
                "image"
            ]  # (12, 224, 224) if ToTensorV2 is used, else (224, 224, 12)

        # Ensure channel first for PyTorch if not already done by transform
        if not isinstance(img_stack, torch.Tensor):
            # Transpose to (C, H, W)
            img_stack = np.transpose(img_stack, (2, 0, 1))
            img_stack = torch.from_numpy(img_stack).float()

        if self.mode == "test":
            return img_stack, subject_id
        else:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_stack, label


# -----------------------------------------------------------------------------
# Transforms & Loader Factory
# -----------------------------------------------------------------------------


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(
    batch_size=32, num_workers=4, load_cached_data=True, debug_limit=None
):
    """
    Factory function to create dataloaders.
    """
    seed_everything(42)

    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    if debug_limit:
        train_df = train_df.head(debug_limit)
        val_df = val_df.head(debug_limit)
        # test_df = test_df.head(debug_limit) # Usually don't limit test unless debugging inference

    # Compute Anchors (using all available data in metadata to ensure coverage)
    # We combine all dfs to compute anchors in one go
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # Deduplicate by ID just in case
    all_df = all_df.drop_duplicates(subset=["BraTS21ID"])

    anchor_dict = compute_anchor_indices(all_df, load_cached_data=load_cached_data)

    # Datasets
    train_ds = MRIDataset(
        train_df, anchor_dict, transform=get_transforms("train"), mode="train"
    )
    val_ds = MRIDataset(
        val_df, anchor_dict, transform=get_transforms("val"), mode="val"
    )
    test_ds = MRIDataset(
        test_df, anchor_dict, transform=get_transforms("test"), mode="test"
    )

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
