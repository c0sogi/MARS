import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.dicom_processing import read_dicom_robust, preprocess_image
from library.roi_selection import get_roi_indices, extract_number

# ------------------------------------------------------------------------------
# Helper Functions for Data Preparation
# ------------------------------------------------------------------------------


def load_slice_data(subject_row, modality, center_slice_idx):
    """
    Loads 3 slices (center - stride, center, center + stride) for a specific modality.
    Returns a numpy array of shape (3, H, W).
    """
    path_col = f"path_{modality}"
    if path_col not in subject_row:
        # Return zeros if path missing
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    dir_path = os.path.join(Config.INPUT_DIR, subject_row[path_col])
    if not os.path.exists(dir_path):
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(".dcm")], key=extract_number
    )
    num_files = len(files)

    if num_files == 0:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Calculate indices with clamping
    # Order: [Previous, Center, Next]
    indices = [
        center_slice_idx - Config.STRIDE,
        center_slice_idx,
        center_slice_idx + Config.STRIDE,
    ]

    slices = []
    for idx in indices:
        # Clamp index to valid range
        idx = max(0, min(idx, num_files - 1))
        file_path = os.path.join(dir_path, files[idx])

        # Read and preprocess (returns float32, resized to Config.IMG_SIZE)
        img = read_dicom_robust(file_path)
        img = preprocess_image(img)

        # Conservative Min-Max Scaling per slice [0, 1]
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        slices.append(img)

    return np.stack(slices, axis=0)  # (3, H, W)


def build_subject_volume(subject_row, roi_row, view_type):
    """
    Builds the 12-channel volume for a specific view.

    Args:
        subject_row: Row from metadata DataFrame.
        roi_row: Row from ROI DataFrame.
        view_type: 1 (Anatomical/FLAIR anchor) or 2 (Pathological/T1wCE anchor).

    Returns:
        np.ndarray: Volume of shape (12, H, W).
    """
    # Determine which anchor index to use
    if view_type == 1:
        anchor_idx = roi_row["roi_anchor1_idx"]
        anchor_modality = Config.ANCHOR_1_MODALITY
    else:
        anchor_idx = roi_row["roi_anchor2_idx"]
        anchor_modality = Config.ANCHOR_2_MODALITY

    # Get slice count of the anchor modality to calculate relative depth
    path_col_anchor = f"path_{anchor_modality}"
    len_anchor = 1
    if path_col_anchor in subject_row:
        p = os.path.join(Config.INPUT_DIR, subject_row[path_col_anchor])
        if os.path.exists(p):
            files_anchor = [f for f in os.listdir(p) if f.endswith(".dcm")]
            len_anchor = len(files_anchor)

    # Avoid division by zero
    len_anchor = max(len_anchor, 1)
    relative_depth = anchor_idx / len_anchor

    # Modality order for Grouped Conv: FLAIR, T1w, T1wCE, T2w
    # This order MUST be consistent for the model stem to work correctly.
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    volume_parts = []
    for mod in modalities:
        # Determine target index in this modality using relative depth
        path_col_mod = f"path_{mod}"
        len_mod = 0
        if path_col_mod in subject_row:
            p = os.path.join(Config.INPUT_DIR, subject_row[path_col_mod])
            if os.path.exists(p):
                files_mod = [f for f in os.listdir(p) if f.endswith(".dcm")]
                len_mod = len(files_mod)

        if len_mod == 0:
            # Missing modality, fill with zeros
            mod_stack = np.zeros(
                (3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )
        else:
            # Map depth to index
            target_idx = int(relative_depth * len_mod)
            mod_stack = load_slice_data(subject_row, mod, target_idx)

        volume_parts.append(mod_stack)

    # Stack: 4 parts * 3 slices = 12 channels
    full_volume = np.concatenate(volume_parts, axis=0)
    return full_volume


def process_dataset(metadata_df, split_name, load_cached_data=True):
    """
    Generates or loads the full dataset array.

    Returns:
        data_array: (N_subjects, 2_views, 12_channels, H, W)
        labels_array: (N_subjects,)
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"data_cache_{split_name}.npy")
    label_cache_path = os.path.join(
        Config.WORKING_DIR, f"labels_cache_{split_name}.npy"
    )

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_path)
        and os.path.exists(label_cache_path)
    ):
        print(f"Loading cached data for {split_name} from {cache_path}...")
        try:
            data = np.load(cache_path)
            labels = np.load(label_cache_path)
            return data, labels
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing data for {split_name} (Dual-Anchor Strategy)...")

    # Get ROI indices (this handles its own caching)
    roi_df = get_roi_indices(metadata_df, split_name, load_cached_data=load_cached_data)

    # Merge roi_df with metadata_df to align rows
    merged_df = pd.merge(metadata_df, roi_df, on="BraTS21ID", how="left")

    n_subjects = len(merged_df)
    data_shape = (n_subjects, 2, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)

    data_array = np.zeros(data_shape, dtype=np.float32)
    labels_list = []

    for i, row in merged_df.iterrows():
        # View 1: Anatomical Anchor
        vol1 = build_subject_volume(row, row, view_type=1)
        data_array[i, 0] = vol1

        # View 2: Pathological Anchor
        vol2 = build_subject_volume(row, row, view_type=2)
        data_array[i, 1] = vol2

        # Label
        if "MGMT_value" in row:
            labels_list.append(row["MGMT_value"])
        else:
            labels_list.append(-1.0)  # Placeholder for test set

    labels_array = np.array(labels_list, dtype=np.float32)

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    np.save(cache_path, data_array)
    np.save(label_cache_path, labels_array)

    return data_array, labels_array


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class BraTSDataset(Dataset):
    def __init__(self, data_array, labels_array, ids, mode="train"):
        """
        Args:
            data_array: (N, 2, 12, H, W)
            labels_array: (N,)
            ids: List of BraTS21IDs
            mode: 'train', 'val', or 'test'
        """
        self.data = data_array
        self.labels = labels_array
        self.ids = ids
        self.mode = mode

        # Define Augmentations
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Rotate with zero padding (constant)
                    A.Rotate(
                        limit=Config.ROTATION_DEGREES,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        # For training, we treat the two views as independent samples -> 2 * N
        if self.mode == "train":
            return len(self.data) * 2
        # For val/test, we return the subject with both views -> N
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Map flat index to (subject, view)
            subject_idx = idx // 2
            view_idx = idx % 2

            # Extract specific view: (12, H, W)
            img_tensor = self.data[subject_idx, view_idx]
            label = self.labels[subject_idx]

            # Albumentations requires (H, W, C)
            img_np = np.transpose(img_tensor, (1, 2, 0))

            # Apply augmentation
            augmented = self.transform(image=img_np)["image"]

            return augmented, torch.tensor(label, dtype=torch.float32)

        else:
            # Val/Test: Return both views for ensemble consensus
            subject_idx = idx

            view1 = self.data[subject_idx, 0]
            view2 = self.data[subject_idx, 1]
            label = self.labels[subject_idx]
            subject_id = self.ids[subject_idx]

            # Transform View 1
            v1_np = np.transpose(view1, (1, 2, 0))
            v1_aug = self.transform(image=v1_np)["image"]

            # Transform View 2
            v2_np = np.transpose(view2, (1, 2, 0))
            v2_aug = self.transform(image=v2_np)["image"]

            return v1_aug, v2_aug, torch.tensor(label, dtype=torch.float32), subject_id


# ------------------------------------------------------------------------------
# Data Loader Factory
# ------------------------------------------------------------------------------


def get_dataloader(
    split_name, batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split_name: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle
        load_cached_data: Whether to use cached .npy files
    """
    # 1. Load Metadata
    if split_name == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split_name == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # 2. Process/Load Data (Heavy Lifting)
    data, labels = process_dataset(df, split_name, load_cached_data=load_cached_data)
    ids = df["BraTS21ID"].values

    # 3. Determine Dataset Mode
    # 'train' split uses 'train' mode (flattened views)
    # 'val' and 'test' splits use 'val'/'test' mode (grouped views)
    mode = (
        "train"
        if split_name == "train"
        else ("test" if split_name == "test" else "val")
    )

    # 4. Create Dataset
    dataset = BraTSDataset(data, labels, ids, mode=mode)

    # 5. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
