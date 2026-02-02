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
CACHE_DIR = "./working/roi_cache_v2"
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
    Computes the anchor slice index for each subject based on the Maximum Sum of Intensity
    in the FLAIR modality. (Cite Lesson 38, Lesson 53)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "roi_cache.parquet")

    # 1. Load Cache if requested
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            return dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Computing ROI anchors (Max Sum Intensity on FLAIR)...")
    anchor_dict = {}

    for idx, row in metadata_df.iterrows():
        subject_id = row["BraTS21ID"]
        flair_dir = os.path.join(INPUT_DIR, row["path_FLAIR"])
        flair_files = get_sorted_files(flair_dir)

        if not flair_files:
            anchor_dict[subject_id] = 0
            continue

        max_energy = -1
        best_idx = 0

        # Scan all FLAIR slices to find the one with maximum signal (brain tissue)
        for i, f in enumerate(flair_files):
            p = os.path.join(flair_dir, f)
            img = load_image_robust(p)
            energy = np.sum(img)
            if energy > max_energy:
                max_energy = energy
                best_idx = i

        anchor_dict[subject_id] = best_idx

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
                # Use default reflection padding instead of constant padding (Cite Lesson 48)
                A.Rotate(limit=15, p=0.5),
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
