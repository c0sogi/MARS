import os
import re
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import seed_everything

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CACHE_DIR = "./working/idea_13"
ROI_CACHE_FILE = "roi_cache_abs.parquet"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def natural_key(string_):
    """
    Sorts strings containing numbers naturally (Image-1, Image-2, ... Image-10).
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def get_sorted_files(folder_path):
    """
    Returns a naturally sorted list of DICOM files in a directory.
    """
    if not os.path.exists(folder_path):
        return []
    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    files.sort(key=natural_key)
    return files


def load_dicom_slice(path):
    """
    Hierarchical data ingestion:
    1. Try pydicom
    2. Try cv2
    3. Fallback to raw binary tail-read
    """
    # 1. Try pydicom
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img.astype(np.float32)
    except Exception:
        pass

    # 2. Try OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # 3. Raw Binary Tail-Read
    try:
        file_size = os.path.getsize(path)
        # Heuristic: 512x512 uint16 = 524288 bytes, 256x256 uint16 = 131072 bytes
        if file_size >= 524288:
            shape = (512, 512)
            bytes_to_read = 512 * 512 * 2
        elif file_size >= 131072:
            shape = (256, 256)
            bytes_to_read = 256 * 256 * 2
        else:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        with open(path, "rb") as f:
            f.seek(-bytes_to_read, 2)
            buffer = f.read(bytes_to_read)

        img = np.frombuffer(buffer, dtype=np.uint16).reshape(shape)
        return img.astype(np.float32)
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def process_roi_for_subject(subject_id, flair_path, input_dir):
    """
    Calculates the anchor slice index for a subject based on FLAIR integral intensity.
    Returns the absolute slice index (int). Cite solution_lesson_node_00024.
    """
    full_flair_path = os.path.join(input_dir, flair_path)
    files = get_sorted_files(full_flair_path)
    num_files = len(files)

    if num_files == 0:
        return 0

    # Bounds: 15% to 85%
    start_idx = int(num_files * 0.15)
    end_idx = int(num_files * 0.85)

    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_files

    max_integral = -1.0
    best_idx = num_files // 2

    # Iterate through candidate slices
    for i in range(start_idx, end_idx):
        f_path = os.path.join(full_flair_path, files[i])
        img = load_dicom_slice(f_path)

        # Integral (Sum)
        current_integral = np.sum(img)

        if current_integral > max_integral:
            max_integral = current_integral
            best_idx = i

    return best_idx


def generate_roi_cache(df, input_dir, load_cached_data=True):
    """
    Generates or loads a cache mapping BraTS21ID to relative anchor position.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, ROI_CACHE_FILE)

    cache = {}

    # 1. Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            cache = pd.Series(
                cache_df.anchor_index.values, index=cache_df.BraTS21ID
            ).to_dict()
        except Exception:
            cache = {}

    # 2. Compute missing
    ids_to_process = []
    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        if sid not in cache:
            ids_to_process.append(row)

    if ids_to_process:
        # print(f"Computing ROI for {len(ids_to_process)} subjects...")
        for row in ids_to_process:
            sid = row["BraTS21ID"]
            flair_path = row["path_FLAIR"]
            abs_anchor = process_roi_for_subject(sid, flair_path, input_dir)
            cache[sid] = abs_anchor

        # 3. Save Cache
        cache_df = pd.DataFrame(
            list(cache.items()), columns=["BraTS21ID", "anchor_index"]
        )
        cache_df.to_parquet(cache_path)

    return cache


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=0.5, std=0.5), ToTensorV2()])


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class MGMTDataset(Dataset):
    def __init__(self, df, root_dir, roi_cache, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.roi_cache = roi_cache
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        # Use absolute index (Cite solution_lesson_node_00024)
        abs_anchor = self.roi_cache.get(sid, None)
        channels = []

        for mod in self.modalities:
            mod_path = row[f"path_{mod}"]
            full_mod_path = os.path.join(self.root_dir, mod_path)
            files = get_sorted_files(full_mod_path)
            num_files = len(files)

            if num_files == 0:
                for _ in range(3):
                    channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                continue

            # If cache missing (shouldn't happen with generate_roi_cache), fallback to middle
            if abs_anchor is None:
                anchor_idx = num_files // 2
            else:
                anchor_idx = int(abs_anchor)

            # Clamp to valid range for this modality
            anchor_idx = max(0, min(anchor_idx, num_files - 1))

            # Stride 5: Anchor-5, Anchor, Anchor+5
            indices = [anchor_idx - 5, anchor_idx, anchor_idx + 5]

            for i in indices:
                i_clamped = max(0, min(i, num_files - 1))
                f_path = os.path.join(full_mod_path, files[i_clamped])

                img = load_dicom_slice(f_path)

                # Resize (Area interpolation for downsampling/noise reduction)
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )

                # Min-Max Normalize to [0, 1]
                mx = img.max()
                if mx > 0:
                    img = img / mx

                channels.append(img)

        # Stack: (H, W, 12)
        img_tensor = np.stack(channels, axis=-1)

        if self.transform:
            augmented = self.transform(image=img_tensor)
            img_tensor = augmented["image"]  # (12, H, W)
        else:
            img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1))

        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_tensor, label
        else:
            return img_tensor, sid


# -----------------------------------------------------------------------------
# Data Loaders Factory
# -----------------------------------------------------------------------------


def get_dataloaders(
    train_metadata_path="./metadata/train.csv",
    val_metadata_path="./metadata/val.csv",
    test_metadata_path="./metadata/test.csv",
    input_dir="./input",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
):
    df_train = pd.read_csv(train_metadata_path, dtype={"BraTS21ID": int})
    df_val = pd.read_csv(val_metadata_path, dtype={"BraTS21ID": int})
    df_test = pd.read_csv(test_metadata_path, dtype={"BraTS21ID": int})

    # Combine for cache generation
    all_dfs = pd.concat([df_train, df_val, df_test], ignore_index=True)
    unique_df = all_dfs.drop_duplicates(subset=["BraTS21ID"])

    roi_cache = generate_roi_cache(
        unique_df, input_dir, load_cached_data=load_cached_data
    )

    train_dataset = MGMTDataset(
        df_train, input_dir, roi_cache, transform=get_transforms("train")
    )
    val_dataset = MGMTDataset(
        df_val, input_dir, roi_cache, transform=get_transforms("val")
    )
    test_dataset = MGMTDataset(
        df_test, input_dir, roi_cache, transform=get_transforms("test")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
