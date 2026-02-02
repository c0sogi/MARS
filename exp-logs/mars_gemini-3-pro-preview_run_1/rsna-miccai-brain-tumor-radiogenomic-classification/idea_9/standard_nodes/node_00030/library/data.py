import os
import glob
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything

# ==========================================
# Helper Functions
# ==========================================


def numerical_sort_key(s):
    """
    Extracts the number from filenames like 'Image-123.dcm' for proper sorting.
    """
    numbers = re.findall(r"\d+", s)
    return int(numbers[-1]) if numbers else -1


def load_dicom_image(path):
    """
    Reads a DICOM file and returns the pixel array.
    Tries pydicom first, falls back to cv2.
    """
    try:
        ds = pydicom.dcmread(path)
        return ds.pixel_array
    except:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        except:
            pass
    return None


def get_brain_voi_indices(file_paths):
    """
    Scans a list of DICOM paths to find the start and end indices of the brain tissue.
    1. Loads middle slice to determine threshold.
    2. Scans from start and end to find brain boundaries.
    """
    if not file_paths:
        return 0, 0

    num_files = len(file_paths)
    mid_idx = num_files // 2

    # Load middle slice to determine threshold
    mid_img = load_dicom_image(file_paths[mid_idx])
    if mid_img is None:
        return 0, num_files - 1

    # Threshold: slightly above 0 to ignore pure background
    # Some scans have artifacts, so we use a small heuristic
    threshold = np.max(mid_img) * 0.05
    if threshold < 1:
        threshold = 1

    # Find Start
    start_idx = 0
    # Scan with a stride to save time, then refine?
    # Given 24h budget, we can afford linear scan or stride 5.
    # Let's do stride 5 then refine.
    found_start = False
    for i in range(0, num_files, 5):
        img = load_dicom_image(file_paths[i])
        if img is not None and np.max(img) > threshold:
            # Refine backwards
            for j in range(max(0, i - 5), i):
                img_ref = load_dicom_image(file_paths[j])
                if img_ref is not None and np.max(img_ref) > threshold:
                    start_idx = j
                    found_start = True
                    break
            if not found_start:
                start_idx = i
                found_start = True
            break

    if not found_start:
        return 0, num_files - 1

    # Find End
    end_idx = num_files - 1
    found_end = False
    for i in range(num_files - 1, -1, -5):
        img = load_dicom_image(file_paths[i])
        if img is not None and np.max(img) > threshold:
            # Refine forwards
            for j in range(min(num_files - 1, i + 5), i, -1):
                img_ref = load_dicom_image(file_paths[j])
                if img_ref is not None and np.max(img_ref) > threshold:
                    end_idx = j
                    found_end = True
                    break
            if not found_end:
                end_idx = i
                found_end = True
            break

    if not found_end:
        end_idx = num_files - 1

    # Safety check
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = num_files - 1

    return start_idx, end_idx


def prepare_data(metadata_path, cache_path, load_cached_data=True, debug_size=None):
    """
    Prepares the dataset by expanding subjects into multiple slice samples.
    Implements caching to avoid re-scanning file systems.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        if debug_size:
            df = df.iloc[: debug_size * 3]  # Approx adjustment since we triple data
        return df

    print(f"Generating data cache for {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if debug_size:
        df_meta = df_meta.iloc[:debug_size]

    expanded_data = []

    # Modalities to process
    modalities = ["flair", "t1wce", "t2w"]

    for _, row in df_meta.iterrows():
        subject_id = row["BraTS21ID"]
        label = row["MGMT_value"] if "MGMT_value" in row else -1

        # Get sorted file lists for each modality
        modality_files = {}
        voi_bounds = {}

        valid_subject = True

        for mod in modalities:
            dir_path = os.path.join(Config.INPUT_DIR, row[f"{mod}_path"])
            if not os.path.exists(dir_path):
                valid_subject = False
                break

            files = glob.glob(os.path.join(dir_path, "*.dcm"))
            files.sort(key=numerical_sort_key)

            if not files:
                valid_subject = False
                break

            modality_files[mod] = files

            # Use full file range (Geometric Prior) instead of unsupervised VOI heuristic
            # Cite solution_lesson_node_00006
            voi_bounds[mod] = (0, len(files) - 1)

        if not valid_subject:
            continue

        # Deterministic Expansion
        # Create 3 samples for this subject
        for depth in Config.SAMPLING_DEPTHS:  # [0.45, 0.50, 0.55]
            sample_entry = {
                "BraTS21ID": subject_id,
                "MGMT_value": label,
                "depth_id": depth,
            }

            # Select slice for each modality based on its own VOI
            for mod in modalities:
                start, end = voi_bounds[mod]
                length = end - start + 1

                # Calculate relative index
                slice_idx = int(start + (length * depth))
                slice_idx = min(max(slice_idx, 0), len(modality_files[mod]) - 1)

                sample_entry[f"{mod}_file"] = modality_files[mod][slice_idx]

            expanded_data.append(sample_entry)

    df_expanded = pd.DataFrame(expanded_data)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_expanded.to_parquet(cache_path, index=False)
    print(f"Saved expanded dataset to {cache_path}. Total samples: {len(df_expanded)}")

    return df_expanded


# ==========================================
# Dataset Class
# ==========================================


class SliceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load 3 channels
        # Channel 1: FLAIR, Channel 2: T1wCE, Channel 3: T2w
        paths = [row["flair_file"], row["t1wce_file"], row["t2w_file"]]
        channels = []

        for p in paths:
            img = load_dicom_image(p)

            if img is None:
                # Fallback for corrupt images: black image
                img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
            else:
                img = img.astype(np.float32)

                # Independent Channel Min-Max Scaling
                min_val = np.min(img)
                max_val = np.max(img)
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)  # Avoid div by zero

            # Resize immediately to save memory/compute before stacking
            # Note: Albumentations resize is usually part of transform,
            # but we need consistent size for stacking if raw images differ.
            # We'll use cv2 for basic resizing to target size first.
            if img.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
                img = cv2.resize(
                    img,
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )

            channels.append(img)

        # Stack: (H, W, 3)
        image = np.stack(channels, axis=-1)

        # Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        label = torch.tensor(row["MGMT_value"], dtype=torch.float)

        return image, label


# ==========================================
# Data Loading Functions
# ==========================================


def get_transforms(data="train"):
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Elastic and Grid as per Idea 9
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose([A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE), ToTensorV2()])


def get_dataloaders(debug_sample_size=None, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    seed_everything(Config.SEED)

    # 1. Prepare DataFrames (with Caching)
    train_df = prepare_data(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_PATH,
        load_cached_data=load_cached_data,
        debug_size=debug_sample_size,
    )

    val_df = prepare_data(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_PATH,
        load_cached_data=load_cached_data,
        debug_size=debug_sample_size,
    )

    test_df = prepare_data(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_PATH,
        load_cached_data=load_cached_data,
        debug_size=debug_sample_size,
    )

    # 2. Create Datasets
    train_dataset = SliceDataset(train_df, transform=get_transforms("train"))
    val_dataset = SliceDataset(val_df, transform=get_transforms("valid"))
    test_dataset = SliceDataset(test_df, transform=get_transforms("test"))

    # 3. Create DataLoaders
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

    return train_loader, val_loader, test_loader, test_df
