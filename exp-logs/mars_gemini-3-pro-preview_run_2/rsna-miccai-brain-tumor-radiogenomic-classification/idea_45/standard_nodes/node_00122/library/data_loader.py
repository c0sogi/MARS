import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Attempt to import pydicom (Cite debug_lesson_9)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from library.config import (
    SEED,
    INPUT_DIR,
    METADATA_TRAIN,
    METADATA_VAL,
    METADATA_TEST,
    IMG_SIZE,
    ROI_DEPTH_MIN,
    ROI_DEPTH_MAX,
    MODALITY_ORDER,
    STRIDES,
    BATCH_SIZE,
    NUM_WORKERS,
    CACHE_DIR,
    DEVICE,
)
from library.utils import seed_everything, setup_logger, ensure_dir

# Initialize Logger
logger = setup_logger("data_loader")


def load_dicom_robust(path):
    """
    Loads a DICOM file with a fallback to raw binary reading.
    Resizes to IMG_SIZE using Area Interpolation.
    Returns a float32 numpy array.
    """
    if not os.path.exists(path):
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    img = None

    # Attempt 0: Pydicom (Cite debug_lesson_9)
    if HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
        except Exception:
            img = None

    # Attempt 1: OpenCV
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Handle 16-bit images potentially read as 3-channel
                if len(img.shape) == 3:
                    img = img[:, :, 0]
        except Exception:
            img = None

    # Attempt 2: Raw Binary Tail-Read
    if img is None:
        try:
            file_size = os.path.getsize(path)

            # Dynamic Resolution Check (Cite debug_lesson_5)
            resolutions = [512, 256, 240, 192, 160, 128]

            for dim in resolutions:
                try:
                    offset = dim * dim * 2
                    if file_size < offset:
                        continue

                    with open(path, "rb") as f:
                        f.seek(-offset, os.SEEK_END)
                        buffer = f.read(offset)

                        if len(buffer) == offset:
                            img = np.frombuffer(buffer, dtype=np.uint16).reshape(
                                (dim, dim)
                            )
                            break
                except Exception:
                    continue

            # Fallback if loop failed
            if img is None:
                dim = int(np.sqrt(file_size // 2))
                offset = dim * dim * 2
                with open(path, "rb") as f:
                    f.seek(-offset, os.SEEK_END)
                    buffer = f.read(offset)
                    img = np.frombuffer(buffer, dtype=np.uint16).reshape((dim, dim))

        except Exception:
            # Final fallback: return zeros
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Resize
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32)


def get_file_id(filename):
    """Extracts the integer ID from 'Image-XXX.dcm'."""
    try:
        # Assumes format Image-123.dcm
        return int(filename.split("-")[1].split(".")[0])
    except Exception:
        return -1


def get_anchor_slice_ids(df, cache_path, load_cached_data=True):
    """
    Determines the anchor slice ID (max intensity in FLAIR) for each subject.
    Implements caching to satisfy deterministic processing requirements.
    """
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert to dictionary mapping BraTS21ID to anchor_id
            return dict(zip(cache_df["BraTS21ID"], cache_df["anchor_id"]))
        except Exception as e:
            logger.warning(f"Failed to load anchor cache: {e}. Recomputing...")

    logger.info("Computing anchor slices for dataset...")
    anchors = {}

    # Iterate over unique subjects in the dataframe
    # We use a set of IDs to avoid redundant processing if df has duplicates (unlikely here)
    subjects = df[["BraTS21ID", "path_FLAIR"]].drop_duplicates()

    for _, row in subjects.iterrows():
        subject_id = row["BraTS21ID"]
        flair_dir = os.path.join(INPUT_DIR, row["path_FLAIR"])

        if not os.path.exists(flair_dir):
            anchors[subject_id] = 0
            continue

        files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]
        if not files:
            anchors[subject_id] = 0
            continue

        # Sort files by ID
        file_map = []  # (id, filename)
        for f in files:
            fid = get_file_id(f)
            if fid != -1:
                file_map.append((fid, f))

        file_map.sort(key=lambda x: x[0])

        if not file_map:
            anchors[subject_id] = 0
            continue

        # Filter by depth (15% - 85%)
        num_slices = len(file_map)
        start_idx = int(num_slices * ROI_DEPTH_MIN)
        end_idx = int(num_slices * ROI_DEPTH_MAX)

        # Ensure at least one slice is checked
        if start_idx == end_idx:
            end_idx = start_idx + 1

        roi_files = file_map[start_idx:end_idx]

        max_intensity = -1.0
        best_id = file_map[num_slices // 2][0]  # Default to middle

        for fid, fname in roi_files:
            path = os.path.join(flair_dir, fname)
            img = load_dicom_robust(path)
            intensity = np.sum(img)

            if intensity > max_intensity:
                max_intensity = intensity
                best_id = fid

        anchors[subject_id] = best_id

    # Save to cache
    ensure_dir(os.path.dirname(cache_path))
    cache_df = pd.DataFrame(list(anchors.items()), columns=["BraTS21ID", "anchor_id"])
    cache_df.to_parquet(cache_path)

    return anchors


def construct_adaptive_volume(row, anchor_id):
    """
    Constructs the 12-channel input tensor using biologically-adaptive strides.
    """
    channels = []

    # Define the geometry of the volume
    # Groups: FLAIR, T2w, T1w, T1wCE

    for mod in MODALITY_ORDER:
        base_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])
        stride = STRIDES[mod]

        # Target IDs: [Anchor-Stride, Anchor, Anchor+Stride]
        target_ids = [anchor_id - stride, anchor_id, anchor_id + stride]

        # We need to know the min/max ID available for this modality to clamp correctly
        # Listing files is expensive, but necessary for correct clamping if we don't cache metadata.
        # Optimization: We assume IDs are somewhat contiguous or we just check existence.
        # However, "Edge Clamping" requires knowing the boundary.
        # Given the "on-the-fly" constraint, we'll implement a soft clamp:
        # If file doesn't exist, we try to find the nearest existing file?
        # Or simpler: We just check existence. If missing, we zero pad.
        # Wait, the prompt explicitly asks for "Edge Clamping" for Z-axis bounds.
        # To do this efficiently without listing dir every time, we can try to load.
        # If load fails, we assume it's out of bounds? No, it could be a gap.
        # Let's list files once per modality.

        if os.path.exists(base_path):
            files = os.listdir(base_path)
            ids = sorted([get_file_id(f) for f in files if f.endswith(".dcm")])
            if not ids:
                # Modality empty
                for _ in range(3):
                    channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                continue

            min_id, max_id = ids[0], ids[-1]

            for tid in target_ids:
                # Spatial Clamping
                clamped_id = tid
                if clamped_id < min_id:
                    clamped_id = min_id
                elif clamped_id > max_id:
                    clamped_id = max_id

                # Construct path
                # Note: Filenames might not be strictly Image-{ID}.dcm if there are prefix variations,
                # but the dataset description shows "Image-1.dcm".
                img_path = os.path.join(base_path, f"Image-{clamped_id}.dcm")

                # Load
                img = load_dicom_robust(img_path)

                # Normalize Independent Per-Channel Min-Max
                if np.max(img) > np.min(img):
                    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                else:
                    img = np.zeros_like(img)  # Avoid division by zero

                channels.append(img)
        else:
            # Missing Modality -> Zero Padding
            for _ in range(3):
                channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))

    # Stack channels: (12, 224, 224)
    volume = np.stack(channels, axis=0)
    return volume


class BraTSDataset(Dataset):
    def __init__(self, df, anchor_map, transform=None):
        self.df = df
        self.anchor_map = anchor_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get Anchor
        anchor_id = self.anchor_map.get(subject_id, 0)

        # Construct Volume
        # Shape: (C, H, W)
        volume = construct_adaptive_volume(row, anchor_id)

        # Apply Transforms
        # Albumentations expects (H, W, C), so we transpose
        if self.transform:
            volume_hwc = np.transpose(volume, (1, 2, 0))
            augmented = self.transform(image=volume_hwc)["image"]
            # ToTensorV2 converts to (C, H, W) and returns Tensor
            volume_tensor = augmented
        else:
            volume_tensor = torch.tensor(volume, dtype=torch.float32)

        # Get Label if available
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return volume_tensor, label
        else:
            return volume_tensor


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(debug=False, max_samples=None):
    """
    Creates DataLoaders for train, val, and test splits.
    Handles anchor caching.
    """
    # Load Metadata
    df_train = pd.read_csv(METADATA_TRAIN)
    df_val = pd.read_csv(METADATA_VAL)
    df_test = pd.read_csv(METADATA_TEST)

    if debug or max_samples:
        limit = max_samples if max_samples else 50
        df_train = df_train.head(limit)
        df_val = df_val.head(limit)
        # Keep test full usually, but for debug we can limit
        if debug:
            df_test = df_test.head(limit)

    # Compute/Load Anchors
    # We combine all DFs to compute anchors in one go if needed, or check cache
    all_dfs = pd.concat([df_train, df_val, df_test], ignore_index=True)
    cache_path = os.path.join(CACHE_DIR, "anchors_cache.parquet")
    anchor_map = get_anchor_slice_ids(all_dfs, cache_path)

    # Create Datasets
    train_dataset = BraTSDataset(
        df_train, anchor_map, transform=get_transforms("train")
    )

    val_dataset = BraTSDataset(df_val, anchor_map, transform=get_transforms("val"))

    test_dataset = BraTSDataset(df_test, anchor_map, transform=get_transforms("test"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
