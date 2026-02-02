import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import (
    load_dicom,
    create_spatial_mask,
    load_or_generate_cache,
    seed_everything,
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def natural_key(string_):
    """
    Key for natural sorting of filenames (e.g., 1.dcm, 2.dcm, 10.dcm).
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def get_study_paths(metadata_df, image_dir):
    """
    Generates a DataFrame containing sorted file paths for each study in metadata.
    """
    path_records = []

    # Get unique study UIDs present in the metadata
    study_uids = metadata_df["StudyInstanceUID"].unique()

    for uid in study_uids:
        study_path = os.path.join(image_dir, uid)
        if not os.path.exists(study_path):
            continue

        # List all .dcm files
        try:
            files = [f for f in os.listdir(study_path) if f.endswith(".dcm")]
        except OSError:
            continue

        # Sort naturally to ensure correct anatomical order (Head to Torso)
        files.sort(key=natural_key)

        for f in files:
            # Extract slice number from filename (e.g., "10.dcm" -> 10)
            # This is crucial for matching with bounding boxes
            try:
                slice_num = int(os.path.splitext(f)[0])
            except ValueError:
                slice_num = -1

            path_records.append(
                {
                    "StudyInstanceUID": uid,
                    "filename": f,
                    "slice_num": slice_num,
                    "rel_path": os.path.join(uid, f),  # Relative to image_dir
                }
            )

    if not path_records:
        return pd.DataFrame(
            columns=["StudyInstanceUID", "filename", "slice_num", "rel_path"]
        )

    return pd.DataFrame(path_records)


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class RSNADataset(Dataset):
    def __init__(self, metadata_df, bbox_df, image_dir, transform=None, phase="train"):
        """
        Args:
            metadata_df: DataFrame with StudyInstanceUID and targets.
            bbox_df: DataFrame with bounding box annotations.
            image_dir: Root directory containing study folders.
            transform: Albumentations transform pipeline.
            phase: 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df
        self.bbox_df = bbox_df
        self.image_dir = image_dir
        self.transform = transform
        self.phase = phase

        # Define cache filename based on phase to separate train/val/test caches
        # This prevents collisions and redundant scanning if subsets differ significantly
        cache_name = f"{phase}_paths_cache.parquet"

        # Define generation function for cache
        def generate_func(df=metadata_df, dir_path=image_dir):
            return get_study_paths(df, dir_path)

        # Load or generate path mapping
        self.paths_df = load_or_generate_cache(
            file_name=cache_name,
            generation_func=generate_func,
            load_cached_data=Config.LOAD_CACHED_DATA,
        )

        # Cite debug_lesson_7: Validate Schema When Loading Cached Data
        # Cite debug_lesson_10: Enforce Explicit Schema on Empty DataFrames
        # Ensure the loaded cache has the required columns. If not, regenerate.
        required_columns = ["StudyInstanceUID", "slice_num", "filename"]

        missing_cols = [c for c in required_columns if c not in self.paths_df.columns]

        # Cite debug_lesson_7: Validate content (not just schema) when loading cached data.
        # If the cache is empty but we have metadata rows, it's likely invalid/stale.
        is_empty = self.paths_df.empty and not self.metadata.empty

        if missing_cols or is_empty:
            reason = (
                f"missing columns {missing_cols}" if missing_cols else "empty content"
            )
            print(f"Cache {cache_name} is invalid ({reason}). Regenerating...")
            self.paths_df = generate_func()

            # Cite debug_lesson_6: Verify Data Generation Output Before Caching
            if self.paths_df.empty and not self.metadata.empty:
                raise RuntimeError(
                    f"Data generation failed: Found 0 images for {len(self.metadata)} metadata entries. "
                    f"Check image paths ({self.image_dir}) and metadata."
                )

            # Attempt to update the cache file to prevent future errors
            try:
                cache_path = os.path.join(Config.CACHE_DIR, cache_name)
                self.paths_df.to_parquet(cache_path, index=False)
            except Exception as e:
                print(f"Warning: Failed to update cache {cache_path}: {e}")

        # Ensure sorted order
        if not self.paths_df.empty:
            self.paths_df = self.paths_df.sort_values(["StudyInstanceUID", "slice_num"])

        # Convert to dictionary for fast O(1) access during __getitem__
        # Key: StudyInstanceUID, Value: List of dicts {'filename':..., 'slice_num':...}
        self.study_map = (
            self.paths_df.groupby("StudyInstanceUID")
            .apply(lambda x: x[["filename", "slice_num"]].to_dict("records"))
            .to_dict()
        )

        # Filter metadata to only include studies we actually found images for
        valid_uids = set(self.study_map.keys())
        self.metadata = self.metadata[
            self.metadata["StudyInstanceUID"].isin(valid_uids)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Get all slices for this study
        slices = self.study_map[uid]
        num_slices = len(slices)

        # --- High-Density Sampling ---
        # Select Config.SEQ_LEN indices uniformly across the Z-axis
        if num_slices >= Config.SEQ_LEN:
            indices = np.linspace(0, num_slices - 1, Config.SEQ_LEN).astype(int)
        else:
            # If fewer slices than required, interpolate indices (nearest neighbor upsampling)
            indices = np.linspace(0, num_slices - 1, Config.SEQ_LEN).astype(int)

        # Prepare containers
        images_tensor = []
        masks_tensor = []
        slice_labels = []

        # Augmentation state (Replay)
        replay_data = None

        for i, slice_idx in enumerate(indices):
            # --- 2.5D Stacking ---
            # Load z-1, z, z+1. Handle boundary conditions by clamping.
            stack_indices = [
                max(0, min(num_slices - 1, z))
                for z in [slice_idx - 1, slice_idx, slice_idx + 1]
            ]

            # Load images
            img_stack = []
            for z_idx in stack_indices:
                s_info = slices[z_idx]
                path = os.path.join(self.image_dir, uid, s_info["filename"])
                # load_dicom handles windowing, normalization, and resizing
                img = load_dicom(path, size=Config.IMAGE_SIZE)
                img_stack.append(img)

            # Stack -> (H, W, 3)
            img_vol = np.stack(img_stack, axis=-1)

            # --- Targets ---
            center_info = slices[slice_idx]
            current_slice_num = center_info["slice_num"]

            # Spatial Mask (H, W)
            mask = create_spatial_mask(
                self.bbox_df, uid, current_slice_num, shape=Config.IMAGE_SIZE
            )

            # Slice Label (Binary): 1 if fracture exists in this slice
            is_fracture = 1.0 if mask.max() > 0.5 else 0.0
            slice_labels.append(is_fracture)

            # --- Synchronized Augmentation ---
            if self.transform:
                if replay_data is None:
                    # First slice: apply transform and save params
                    augmented = self.transform(image=img_vol, mask=mask)
                    replay_data = augmented["replay"]
                    img_aug = augmented["image"]
                    mask_aug = augmented["mask"]
                else:
                    # Subsequent slices: replay transform with exact same params
                    augmented = self.transform.replay(
                        replay_data, image=img_vol, mask=mask
                    )
                    img_aug = augmented["image"]
                    mask_aug = augmented["mask"]
            else:
                # Fallback if no transform provided (shouldn't happen with get_loaders)
                img_aug = torch.from_numpy(img_vol.transpose(2, 0, 1))  # HWC -> CHW
                mask_aug = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW

            images_tensor.append(img_aug)
            masks_tensor.append(mask_aug)

        # Stack sequence
        # images: (SEQ_LEN, 3, H, W)
        # masks: (SEQ_LEN, 1, H, W)
        images_seq = torch.stack(images_tensor)
        masks_seq = torch.stack(masks_tensor)
        slice_labels_seq = torch.tensor(slice_labels, dtype=torch.float32)

        # --- Study Level Targets ---
        # Columns: C1..C7, patient_overall
        if "patient_overall" in row:
            study_targets = row[Config.TARGET_COLS].values.astype(np.float32)
            study_targets = torch.tensor(study_targets)
        else:
            # Test set placeholders
            study_targets = torch.zeros(Config.NUM_CLASSES)

        return {
            "image": images_seq,  # (Seq, 3, H, W)
            "label_study": study_targets,  # (8,)
            "label_slice": slice_labels_seq,  # (Seq,)
            "label_spatial": masks_seq,  # (Seq, 1, H, W)
        }


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------


def get_transforms(phase):
    """
    Returns Albumentations ReplayCompose pipeline.
    ReplayCompose is essential for applying identical transforms across the sequence.
    """
    if phase == "train":
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ]
        )
    else:
        return A.ReplayCompose([ToTensorV2()])


def get_loaders(
    train_meta_path=Config.TRAIN_METADATA,
    val_meta_path=Config.VAL_METADATA,
    bbox_path=Config.BOUNDING_BOXES,
):
    # Load Metadata
    train_df = pd.read_csv(train_meta_path)
    val_df = pd.read_csv(val_meta_path)

    # Load Bounding Boxes
    if os.path.exists(bbox_path):
        bbox_df = pd.read_csv(bbox_path)
    else:
        bbox_df = pd.DataFrame(
            columns=["StudyInstanceUID", "slice_number", "x", "y", "width", "height"]
        )

    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Datasets
    train_dataset = RSNADataset(
        train_df,
        bbox_df,
        Config.TRAIN_IMAGES_DIR,
        transform=train_transform,
        phase="train",
    )

    val_dataset = RSNADataset(
        val_df, bbox_df, Config.TRAIN_IMAGES_DIR, transform=val_transform, phase="val"
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )

    return train_loader, val_loader


def get_test_loader(test_meta_path=Config.TEST_METADATA):
    test_df = pd.read_csv(test_meta_path)
    bbox_df = pd.DataFrame()  # No bboxes for test

    transform = get_transforms("val")

    dataset = RSNADataset(
        test_df, bbox_df, Config.TEST_IMAGES_DIR, transform=transform, phase="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
