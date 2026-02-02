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
    Implements robust partial caching to handle sequential data splits.
    Cite Lesson 00081: Avoid naive "check-and-return" caching logic.
    """
    anchors = {}
    cache_dirty = False

    # 1. Load existing cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            anchors = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_id"]))
            logger.info(f"Loaded {len(anchors)} anchors from cache.")
        except Exception as e:
            logger.warning(f"Failed to load anchor cache: {e}. Starting fresh.")

    # 2. Identify missing subjects
    # We need anchors for every subject in the provided dataframe
    required_ids = df["BraTS21ID"].unique()
    missing_ids = [uid for uid in required_ids if uid not in anchors]

    if not missing_ids:
        return anchors

    logger.info(f"Computing anchors for {len(missing_ids)} missing subjects...")

    # 3. Compute missing anchors
    # Filter DF to only missing subjects to avoid redundant computation
    df_missing = df[df["BraTS21ID"].isin(missing_ids)].drop_duplicates(
        subset=["BraTS21ID"]
    )

    for _, row in df_missing.iterrows():
        subject_id = row["BraTS21ID"]
        flair_dir = os.path.join(INPUT_DIR, row["path_FLAIR"])

        best_id = 0  # Default

        if os.path.exists(flair_dir):
            files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]

            # Map IDs
            file_map = []
            for f in files:
                fid = get_file_id(f)
                if fid != -1:
                    file_map.append((fid, f))

            file_map.sort(key=lambda x: x[0])

            if file_map:
                # Filter by depth (15% - 85%)
                num_slices = len(file_map)
                start_idx = int(num_slices * ROI_DEPTH_MIN)
                end_idx = int(num_slices * ROI_DEPTH_MAX)
                if start_idx == end_idx:
                    end_idx = start_idx + 1

                roi_files = file_map[start_idx:end_idx]

                # Find max intensity
                max_intensity = -1.0
                best_id = file_map[num_slices // 2][0]  # Fallback to middle

                for fid, fname in roi_files:
                    path = os.path.join(flair_dir, fname)
                    img = load_dicom_robust(path)
                    intensity = np.sum(img)

                    if intensity > max_intensity:
                        max_intensity = intensity
                        best_id = fid

        anchors[subject_id] = best_id
        cache_dirty = True

    # 4. Update Cache
    if cache_dirty:
        ensure_dir(os.path.dirname(cache_path))
        cache_df = pd.DataFrame(
            list(anchors.items()), columns=["BraTS21ID", "anchor_id"]
        )
        cache_df.to_parquet(cache_path)
        logger.info(f"Updated anchor cache saved to {cache_path}")

    return anchors


class BraTSDataset(Dataset):
    def __init__(self, df, anchor_map, transform=None):
        self.df = df
        self.anchor_map = anchor_map
        self.transform = transform

        # Pre-scan modality bounds to avoid os.listdir in __getitem__
        # This optimizes I/O throughput significantly.
        self.bounds_cache = {}
        self._scan_bounds()

    def _scan_bounds(self):
        """
        Scans the directory structure to find min/max file IDs for each subject/modality.
        This allows for O(1) clamping during training.
        """
        for _, row in self.df.iterrows():
            subject_id = row["BraTS21ID"]
            self.bounds_cache[subject_id] = {}

            for mod in MODALITY_ORDER:
                dir_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])
                min_id, max_id = 0, 0

                if os.path.exists(dir_path):
                    # We assume standard naming Image-{ID}.dcm
                    # To be fast, we list once.
                    # Note: This adds initialization time but saves training time.
                    try:
                        files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
                        ids = [get_file_id(f) for f in files]
                        ids = [i for i in ids if i != -1]
                        if ids:
                            min_id, max_id = min(ids), max(ids)
                    except OSError:
                        pass

                self.bounds_cache[subject_id][mod] = (min_id, max_id)

    def construct_adaptive_volume(self, row, anchor_id):
        """
        Constructs the 12-channel input tensor using biologically-adaptive strides.
        Uses cached bounds for efficient clamping.
        """
        channels = []
        subject_id = row["BraTS21ID"]

        for mod in MODALITY_ORDER:
            base_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])
            stride = STRIDES[mod]

            # Retrieve cached bounds
            min_id, max_id = self.bounds_cache.get(subject_id, {}).get(mod, (0, 0))

            # If max_id is 0, likely empty or missing
            if max_id == 0:
                for _ in range(3):
                    channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                continue

            # Target IDs: [Anchor-Stride, Anchor, Anchor+Stride]
            target_ids = [anchor_id - stride, anchor_id, anchor_id + stride]

            for tid in target_ids:
                # Spatial Clamping
                clamped_id = tid
                if clamped_id < min_id:
                    clamped_id = min_id
                elif clamped_id > max_id:
                    clamped_id = max_id

                img_path = os.path.join(base_path, f"Image-{clamped_id}.dcm")
                img = load_dicom_robust(img_path)

                # Normalize
                if np.max(img) > np.min(img):
                    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                else:
                    img = np.zeros_like(img)

                channels.append(img)

        volume = np.stack(channels, axis=0)
        return volume

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get Anchor
        anchor_id = self.anchor_map.get(subject_id, 0)

        # Construct Volume
        volume = self.construct_adaptive_volume(row, anchor_id)

        # Apply Transforms
        if self.transform:
            volume_hwc = np.transpose(volume, (1, 2, 0))
            augmented = self.transform(image=volume_hwc)["image"]
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
