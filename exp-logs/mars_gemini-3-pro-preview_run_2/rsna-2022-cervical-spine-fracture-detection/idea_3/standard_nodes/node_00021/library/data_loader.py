import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Helper Functions
# =========================================================================


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., 1.dcm, 2.dcm, 10.dcm).
    Extracts the integer slice number from the filename.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", os.path.basename(s))
    ]


def get_slice_number(filename):
    """Extracts slice number from filename for sorting."""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    if name.isdigit():
        return int(name)
    # Fallback for non-standard names
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return 0


def apply_bone_window(image, intercept, slope):
    """
    Applies bone windowing (WL: 300, WW: 1500) to CT image.
    """
    # Convert to Hounsfield Units (HU)
    image = image * slope + intercept

    # Bone Window settings
    center = 300
    width = 1500

    lower = center - width // 2
    upper = center + width // 2

    # Clip and normalize to [0, 1]
    image = np.clip(image, lower, upper)
    image = (image - lower) / width
    return image.astype(np.float32)


def load_dicom_slice(path, target_size):
    """
    Reads a DICOM file, applies windowing, and resizes.
    Returns a (H, W) float32 array in [0, 1].
    """
    try:
        dcm = pydicom.dcmread(path, stop_before_pixels=False)

        if not hasattr(dcm, "pixel_array"):
            raise ValueError("No pixel_array")

        img = dcm.pixel_array.astype(np.float32)

        # Get Rescale Metadata
        intercept = (
            dcm.RescaleIntercept if hasattr(dcm, "RescaleIntercept") else -1024.0
        )
        slope = dcm.RescaleSlope if hasattr(dcm, "RescaleSlope") else 1.0

        img = apply_bone_window(img, intercept, slope)

        # Resize
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )

        return img

    except Exception as e:
        # Return black image on failure
        return np.zeros((target_size, target_size), dtype=np.float32)


def process_study_paths(metadata_df, root_dir):
    """
    Scans directories to find and sort all slice files for each study.
    Returns a DataFrame suitable for caching.
    """
    records = []

    # Iterate over unique studies in metadata
    unique_studies = metadata_df["StudyInstanceUID"].unique()

    for study_id in unique_studies:
        # Construct path. Metadata image_path is relative to input/
        # But we need to be careful. The metadata script generated 'image_path'
        # as 'train_images/UID'. We need to join with INPUT_DIR.
        # However, we can also just look in root_dir/study_id if root_dir is correct.

        study_dir = os.path.join(root_dir, study_id)

        if not os.path.exists(study_dir):
            continue

        # List all dcm files
        files = glob.glob(os.path.join(study_dir, "*.dcm"))

        # Sort based on slice number
        files.sort(key=natural_sort_key)

        # Store relative paths to save space in cache
        # Relative to root_dir
        for idx, f in enumerate(files):
            rel_path = os.path.relpath(f, root_dir)
            records.append(
                {"StudyInstanceUID": study_id, "rel_path": rel_path, "slice_index": idx}
            )

    return pd.DataFrame(records)


def get_study_file_map(metadata_df, root_dir, cache_path, load_cached_data=True):
    """
    Returns a dict {StudyUID: [list_of_full_paths]} using caching.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    df_cache = None

    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_cache = pd.read_parquet(cache_path)
            # Verify if cache covers the requested studies (basic check)
            # If cache is empty or missing studies, we might want to recompute,
            # but for now we assume cache is valid if it exists.
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}")
            df_cache = None

    # 2. Compute if needed
    if df_cache is None:
        print(f"Computing file paths for {root_dir}...")
        df_cache = process_study_paths(metadata_df, root_dir)
        # Save
        df_cache.to_parquet(cache_path, index=False)

    # 3. Convert to dict for fast access
    # Group by study and collect paths
    # Assuming df_cache is sorted by slice_index implicitly or we sort it
    df_cache = df_cache.sort_values(["StudyInstanceUID", "slice_index"])

    study_map = df_cache.groupby("StudyInstanceUID")["rel_path"].apply(list).to_dict()

    # Convert relative paths back to full paths
    final_map = {}
    for study_id, rel_paths in study_map.items():
        final_map[study_id] = [os.path.join(root_dir, p) for p in rel_paths]

    return final_map


# =========================================================================
# Dataset Class
# =========================================================================


class CervicalSpineDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        root_dir,
        cache_path,
        phase="train",
        transform=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata with StudyInstanceUID and labels.
            root_dir (str): Directory containing study folders.
            cache_path (str): Path to parquet file for caching file lists.
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline.
            load_cached_data (bool): Whether to use cached file paths.
        """
        self.metadata = metadata_df
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        # Load file mapping (Study -> List of sorted slice paths)
        self.study_file_map = get_study_file_map(
            metadata_df, root_dir, cache_path, load_cached_data
        )

        # Filter metadata to only include studies we found files for
        valid_studies = set(self.study_file_map.keys())
        self.metadata = self.metadata[
            self.metadata["StudyInstanceUID"].isin(valid_studies)
        ].reset_index(drop=True)

        self.seq_len = Config.SEQ_LEN
        self.image_size = Config.IMAGE_SIZE

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # Get all slices for this study
        all_files = self.study_file_map[study_uid]
        num_slices = len(all_files)

        # Sampling indices uniformly across the Z-axis
        if num_slices >= self.seq_len:
            indices = np.linspace(0, num_slices - 1, self.seq_len).astype(int)
        else:
            # Padding if fewer slices than seq_len (rare for CT, but possible)
            # We repeat the available slices
            indices = np.linspace(0, num_slices - 1, self.seq_len).astype(int)

        # Load 2.5D stacks
        # Output shape: (SEQ_LEN, H, W, 3)
        sequence_images = []

        for i in indices:
            # Identify neighbors (clamp to boundaries)
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_slices - 1, i + 1)

            path_prev = all_files[idx_prev]
            path_curr = all_files[idx_curr]
            path_next = all_files[idx_next]

            # Load slices
            img_prev = load_dicom_slice(path_prev, self.image_size)
            img_curr = load_dicom_slice(path_curr, self.image_size)
            img_next = load_dicom_slice(path_next, self.image_size)

            # Stack: (H, W, 3)
            img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)
            sequence_images.append(img_stack)

        # Apply Volumetric-Consistent Augmentation
        if self.transform:
            # Use ReplayCompose logic manually or via Albumentations
            # Since we have a list of images, we apply the transform to the first
            # and replay parameters for the rest.

            # 1. Apply to first image
            res = self.transform(image=sequence_images[0])
            augmented_sequence = [res["image"]]
            replay_params = res["replay"]

            # 2. Replay for the rest
            for i in range(1, len(sequence_images)):
                res = A.ReplayCompose.replay(replay_params, image=sequence_images[i])
                augmented_sequence.append(res["image"])

            sequence_images = augmented_sequence

        # Convert to Tensor: (SEQ_LEN, H, W, 3) -> (SEQ_LEN, 3, H, W)
        # Assuming transform output is numpy array or tensor.
        # If ToTensorV2 was used, it's tensor (3, H, W).
        # If not, it's numpy (H, W, 3).

        # Check type of first element
        if isinstance(sequence_images[0], torch.Tensor):
            # Stack tensors: (SEQ_LEN, 3, H, W)
            data_tensor = torch.stack(sequence_images)
        else:
            # Stack numpy: (SEQ_LEN, H, W, 3)
            data_numpy = np.stack(sequence_images)
            # To Tensor: (SEQ_LEN, 3, H, W)
            data_tensor = torch.from_numpy(data_numpy).permute(0, 3, 1, 2)

        # Prepare Targets
        if self.phase != "test":
            # Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
            # Note: Config.NUM_CLASSES is 8.
            # We map columns to this specific order.

            labels = [
                row["C1"],
                row["C2"],
                row["C3"],
                row["C4"],
                row["C5"],
                row["C6"],
                row["C7"],
                row["patient_overall"],
            ]
            target_tensor = torch.tensor(labels, dtype=torch.float32)

            return {
                "image": data_tensor,
                "target": target_tensor,
                "study_id": study_uid,
            }
        else:
            return {"image": data_tensor, "study_id": study_uid}


# =========================================================================
# Data Loaders
# =========================================================================


def get_transforms(phase):
    """
    Returns Albumentations transforms.
    Uses ReplayCompose to support consistent augmentation across the sequence.
    """
    if phase == "train":
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
                # Normalize is not strictly needed as we did manual [0,1] normalization in load_dicom
                # But standardizing mean/std is good for EfficientNet.
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.ReplayCompose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached file paths.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debugging: Subsample if needed
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Create Datasets
    train_dataset = CervicalSpineDataset(
        train_df,
        Config.TRAIN_IMAGES_DIR,
        Config.TRAIN_CACHE_PATH,
        phase="train",
        transform=get_transforms("train"),
        load_cached_data=load_cached_data,
    )

    val_dataset = CervicalSpineDataset(
        val_df,
        Config.TRAIN_IMAGES_DIR,  # Validation images are in train_images folder
        Config.VAL_CACHE_PATH,
        phase="val",
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
    )

    test_dataset = CervicalSpineDataset(
        test_df,
        Config.TEST_IMAGES_DIR,
        Config.TEST_CACHE_PATH,
        phase="test",
        transform=get_transforms("test"),
        load_cached_data=load_cached_data,
    )

    # 3. Create Loaders
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
