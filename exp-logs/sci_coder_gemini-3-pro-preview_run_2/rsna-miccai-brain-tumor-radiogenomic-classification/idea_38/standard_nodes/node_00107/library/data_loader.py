import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from library.config import Config


# ------------------------------------------------------------------------------
# 1. Robust Image Loading
# ------------------------------------------------------------------------------
def load_image_robust(path):
    """
    Reads DICOMs using OpenCV with a fallback to raw binary tail-reading for
    corrupt files, ensuring float32 conversion and Area Interpolation resizing.
    """
    img = None

    # Method 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Method 2: Raw Binary Tail-Read REMOVED
    # Cite Lesson 00101: Prioritize established libraries over custom parsers.
    # Avoiding "garbage-in" from compressed DICOMs that raw reading misinterprets.

    # Fallback: Return zeros if OpenCV fails
    if img is None:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Ensure float32
    img = img.astype(np.float32)

    # Resize to 224x224 using Area Interpolation
    if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
        img = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )

    return img


# ------------------------------------------------------------------------------
# 2. Anchor Calculation
# ------------------------------------------------------------------------------
def compute_flair_anchor(flair_dir):
    """
    Scans a subject's FLAIR directory to calculate the sum of pixel intensities
    for each slice (raw values, no smoothing) within the 15-85% depth range,
    returning the index of the slice with the maximum integral.
    """
    # List all DICOM files, sort by ID (filename number)
    files = sorted(
        glob.glob(os.path.join(flair_dir, "*.dcm")),
        key=lambda x: int(os.path.basename(x).split("-")[-1].split(".")[0]),
    )

    n_files = len(files)
    if n_files == 0:
        return 0, []

    # Define range 15% - 85%
    start_idx = int(n_files * 0.15)
    end_idx = int(n_files * 0.85)

    # Handle small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n_files

    max_integral = -1.0
    anchor_idx = n_files // 2  # Default to middle

    # Iterate and compute integral
    for i in range(start_idx, end_idx):
        path = files[i]
        img = load_image_robust(path)
        integral = np.sum(img)

        if integral > max_integral:
            max_integral = integral
            anchor_idx = i

    return anchor_idx, files


# ------------------------------------------------------------------------------
# 3. Volume Loading with Strict Geometric Alignment
# ------------------------------------------------------------------------------
def load_subject_volume(subject_id, metadata_df):
    """
    Loads volume data implementing the Strict Geometric Data Pipeline.
    """
    row = metadata_df[metadata_df["BraTS21ID"] == subject_id].iloc[0]

    path_flair = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
    path_t1w = os.path.join(Config.INPUT_DIR, row["path_T1w"])
    path_t1wce = os.path.join(Config.INPUT_DIR, row["path_T1wCE"])
    path_t2w = os.path.join(Config.INPUT_DIR, row["path_T2w"])

    # 1. FLAIR Anchor
    anchor_idx, flair_files = compute_flair_anchor(path_flair)

    if not flair_files:
        return np.zeros(
            (Config.CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    n_flair = len(flair_files)

    # 2. Define Target Indices (Spatial Neighbors: [Anchor-5, Anchor, Anchor+5])
    offsets = [-Config.STRIDE, 0, Config.STRIDE]
    target_indices = [anchor_idx + o for o in offsets]

    volume_channels = []
    modality_paths = [path_flair, path_t1w, path_t1wce, path_t2w]

    # Determine File IDs for the chosen slices based on FLAIR
    target_file_ids = []

    for t_idx in target_indices:
        # Spatial Continuity: Edge Clamping on Index for FLAIR
        clamped_idx = max(0, min(n_flair - 1, t_idx))

        # Get the file ID from the FLAIR list at the clamped index
        f_path = flair_files[clamped_idx]
        try:
            fid = int(os.path.basename(f_path).split("-")[-1].split(".")[0])
        except:
            fid = -1
        target_file_ids.append(fid)

    # Load for each modality
    for mod_path in modality_paths:
        mod_slices = []
        for fid in target_file_ids:
            # Construct path: Image-{fid}.dcm
            img_path = os.path.join(mod_path, f"Image-{fid}.dcm")

            # Cross-Modality Alignment: Check existence
            if os.path.exists(img_path):
                img = load_image_robust(img_path)
            else:
                # Zero Padding (Blank Image) - Do NOT Clamp
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

            # Normalization: Independent Per-Channel Min-Max Scaling [0, 1]
            if np.max(img) > np.min(img):
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
            else:
                img = np.zeros_like(img)

            mod_slices.append(img)

        volume_channels.extend(mod_slices)

    # Stack -> (12, 224, 224)
    volume = np.array(volume_channels, dtype=np.float32)
    return volume


# ------------------------------------------------------------------------------
# 4. Dataset Processing & Caching
# ------------------------------------------------------------------------------
def process_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Processes dataset with caching mechanism.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path_data = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npy")
    cache_path_labels = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")

    # 1. Try Load
    if (
        load_cached_data
        and os.path.exists(cache_path_data)
        and os.path.exists(cache_path_labels)
    ):
        print(f"Loading {cache_name} from cache...")
        try:
            data = np.load(cache_path_data)
            labels = np.load(cache_path_labels)
            return data, labels
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute
    print(f"Processing {cache_name} from scratch...")
    df = pd.read_csv(metadata_path)

    data_list = []
    labels_list = []

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        try:
            vol = load_subject_volume(sid, df)
            data_list.append(vol)

            if "MGMT_value" in row:
                labels_list.append(row["MGMT_value"])
            else:
                labels_list.append(0.5)  # Dummy for test
        except Exception as e:
            print(f"Error processing subject {sid}: {e}")
            data_list.append(
                np.zeros(
                    (Config.CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                    dtype=np.float32,
                )
            )
            labels_list.append(0.5)

    data = np.array(data_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    # 3. Save
    np.save(cache_path_data, data)
    np.save(cache_path_labels, labels)

    return data, labels


# ------------------------------------------------------------------------------
# 5. Dataset Class
# ------------------------------------------------------------------------------
class BraTSDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # (12, H, W)
        y = self.labels[idx]

        # Augmentations (Geometric only)
        if self.transform:
            # Convert to (H, W, C) for OpenCV
            x_np = x.transpose(1, 2, 0)

            # Horizontal Flip
            if np.random.rand() > 0.5:
                x_np = cv2.flip(x_np, 1)

            # Vertical Flip
            if np.random.rand() > 0.5:
                x_np = cv2.flip(x_np, 0)

            # Rotation +/- 15 deg with Reflection Padding
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = x_np.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                x_np = cv2.warpAffine(x_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Convert back to (C, H, W)
            x = x_np.transpose(2, 0, 1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )
