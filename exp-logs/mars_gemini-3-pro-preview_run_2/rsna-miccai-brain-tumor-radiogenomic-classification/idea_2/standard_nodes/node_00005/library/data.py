import os
import re
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import log_message

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class DicomReader:
    """
    Robust DICOM reader that attempts multiple methods to read pixel data:
    1. pydicom (if available)
    2. OpenCV (if supported)
    3. Raw binary fallback (assuming 512x512 or 256x256 uint16)
    """

    @staticmethod
    def read_file(path, target_size=None):
        """
        Reads a DICOM file and returns a normalized numpy array (0-1).

        Args:
            path (str): Path to the .dcm file.
            target_size (int, optional): If set, resizes the image to (target_size, target_size).

        Returns:
            np.ndarray: Image array of shape (H, W) with values in [0, 1].
        """
        img = None

        # Method 1: pydicom
        if HAS_PYDICOM:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)
            except Exception:
                pass

        # Method 2: OpenCV
        if img is None:
            try:
                # cv2.imread usually fails on standard DICOMs but works if they are just renamed images
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = img.astype(np.float32)
            except Exception:
                pass

        # Method 3: Raw Binary Fallback
        if img is None:
            try:
                with open(path, "rb") as f:
                    data = f.read()

                # Heuristic: Check file size to guess resolution (uint16 = 2 bytes per pixel)
                # 512x512
                size_512 = 512 * 512 * 2
                # 256x256
                size_256 = 256 * 256 * 2

                if len(data) >= size_512:
                    # Assume data is at the end
                    pixel_data = data[-size_512:]
                    img = (
                        np.frombuffer(pixel_data, dtype=np.uint16)
                        .reshape(512, 512)
                        .astype(np.float32)
                    )
                elif len(data) >= size_256:
                    pixel_data = data[-size_256:]
                    img = (
                        np.frombuffer(pixel_data, dtype=np.uint16)
                        .reshape(256, 256)
                        .astype(np.float32)
                    )
            except Exception:
                pass

        # Fallback if everything fails: Return black image
        if img is None:
            # Default to target size or 224
            sz = target_size if target_size else 224
            return np.zeros((sz, sz), dtype=np.float32)

        # Handle dimensions (sometimes DICOMs are 3D or have extra dims)
        if img.ndim > 2:
            img = img[0]  # Take first channel/slice if multiple exist

        # Resize if requested
        if target_size is not None and (
            img.shape[0] != target_size or img.shape[1] != target_size
        ):
            img = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )

        # Normalize to 0-1
        # Robust normalization: subtract min, divide by max-min
        # To avoid noise amplifying, we can just divide by max possible (65535) or instance max
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        return img


class ROISelector:
    """
    Identifies the anatomical anchor slice for a subject based on FLAIR signal intensity.
    """

    @staticmethod
    def get_sorted_files(dir_path):
        """Returns sorted list of .dcm files in a directory based on numerical index."""
        if not os.path.exists(dir_path):
            return []

        files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
        # Sort by the number in 'Image-X.dcm'
        files.sort(
            key=lambda x: (
                int(re.search(r"Image-(\d+)", x).group(1))
                if re.search(r"Image-(\d+)", x)
                else 0
            )
        )
        return files

    @staticmethod
    def compute_anchor(subject_path_flair):
        """
        Scans FLAIR slices and returns the index (integer) of the slice with maximum signal intensity.
        Returns middle slice index if directory is empty or unreadable.
        """
        files = ROISelector.get_sorted_files(subject_path_flair)
        if not files:
            return 0

        max_intensity = -1
        best_idx = len(files) // 2  # Default to middle

        # Optimization: We don't need to read every single pixel of every slice fully.
        # We can read every Nth slice to find the region, then refine, or just read all (dataset is small enough).
        # Given the time constraints and small dataset (~500 subjects), reading all is feasible if optimized.
        # To be safe on time, we can skip ends (often empty).

        start = int(len(files) * 0.15)
        end = int(len(files) * 0.85)

        # Map file index to list index
        # The file names are Image-X.dcm. The numbers might not be contiguous or start at 1.
        # We will return the list index (0 to N-1) to be used with the sorted file list.

        for i in range(start, end):
            f_path = os.path.join(subject_path_flair, files[i])
            # Quick read - we can use the binary fallback directly for speed as we just need sum
            try:
                img = DicomReader.read_file(
                    f_path, target_size=64
                )  # Small size for speed
                intensity = np.sum(img)
                if intensity > max_intensity:
                    max_intensity = intensity
                    best_idx = i
            except Exception:
                continue

        return best_idx


def get_roi_map(metadata_df, load_cached_data=True):
    """
    Generates or loads a map of {BraTS21ID: anchor_slice_index}.
    Implements caching using Parquet.
    """
    cache_path = Config.CACHE_PATH

    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            log_message(f"Loading ROI cache from {cache_path}...")
            cache_df = pd.read_parquet(cache_path)
            # Convert to dict
            return dict(zip(cache_df["BraTS21ID"], cache_df["anchor_idx"]))
        except Exception as e:
            log_message(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    log_message("Computing anatomical anchors (ROIs) for all subjects...")
    roi_map = {}

    # We need full paths. Metadata contains relative paths.
    # We iterate unique subjects in the provided dataframe.
    unique_subjects = metadata_df[["BraTS21ID", "path_FLAIR"]].drop_duplicates()

    count = 0
    total = len(unique_subjects)

    for _, row in unique_subjects.iterrows():
        sid = row["BraTS21ID"]
        flair_rel_path = row["path_FLAIR"]
        full_path = os.path.join(Config.INPUT_DIR, flair_rel_path)

        anchor_idx = ROISelector.compute_anchor(full_path)
        roi_map[sid] = anchor_idx

        count += 1
        if count % 50 == 0:
            log_message(f"Processed {count}/{total} subjects...")

    # 3. Save cache
    try:
        cache_df = pd.DataFrame(
            list(roi_map.items()), columns=["BraTS21ID", "anchor_idx"]
        )
        cache_df.to_parquet(cache_path, index=False)
        log_message(f"ROI cache saved to {cache_path}")
    except Exception as e:
        log_message(f"Warning: Could not save cache: {e}")

    return roi_map


class BraTSDataset(Dataset):
    def __init__(self, df, roi_map, transform=None, phase="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            roi_map (dict): Dictionary mapping BraTS21ID to anchor slice index.
            transform (albumentations.Compose): Augmentation pipeline.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.roi_map = roi_map
        self.transform = transform
        self.phase = phase
        self.modalities = Config.MODALITIES  # ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.stride = Config.SLICE_STRIDE
        self.img_size = Config.IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get Anchor
        anchor_idx = self.roi_map.get(subject_id, 0)

        # Define slice indices: [Anchor - Stride, Anchor, Anchor + Stride]
        # We clamp these later based on actual file counts per modality
        relative_indices = [-self.stride, 0, self.stride]

        channels = []

        for mod in self.modalities:
            mod_path_rel = row[f"path_{mod}"]
            mod_dir = os.path.join(Config.INPUT_DIR, mod_path_rel)

            # Get sorted files for this modality
            files = ROISelector.get_sorted_files(mod_dir)
            num_files = len(files)

            if num_files == 0:
                # Missing modality: pad with zeros
                for _ in range(3):
                    channels.append(
                        np.zeros((self.img_size, self.img_size), dtype=np.float32)
                    )
                continue

            # Map anchor (from FLAIR) to this modality
            # Note: Different modalities might have different slice counts.
            # We assume registered images where slice N corresponds roughly to slice N.
            # If counts differ significantly, we normalize index by ratio.
            # For simplicity and robustness (assuming coregistered within reason), we clamp.

            for rel_idx in relative_indices:
                target_idx = anchor_idx + rel_idx

                # Clamp index
                target_idx = max(0, min(target_idx, num_files - 1))

                file_path = os.path.join(mod_dir, files[target_idx])
                img = DicomReader.read_file(file_path, target_size=self.img_size)
                channels.append(img)

        # Stack channels: (H, W, 12)
        # Order: FLAIR(3), T1w(3), T1wCE(3), T2w(3)
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor manually if no transform provided (fallback)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Get Label (if exists)
        label = torch.tensor(0.0)
        if "MGMT_value" in row:
            label = torch.tensor(float(row["MGMT_value"]), dtype=torch.float32)

        return image, label


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # No destructive augmentations (Cutout, Dropout) as per idea
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(debug=False):
    """
    Factory function to create dataloaders for train, val, and test.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        train_df = train_df.iloc[:20]
        val_df = val_df.iloc[:10]

    # 2. Compute/Load ROI Map (Union of all subjects)
    # Concatenate all dfs to ensure we cover everyone
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    roi_map = get_roi_map(all_df, load_cached_data=True)

    # 3. Create Datasets
    train_ds = BraTSDataset(
        train_df, roi_map, transform=get_transforms("train"), phase="train"
    )
    val_ds = BraTSDataset(val_df, roi_map, transform=get_transforms("val"), phase="val")
    test_ds = BraTSDataset(
        test_df, roi_map, transform=get_transforms("test"), phase="test"
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
