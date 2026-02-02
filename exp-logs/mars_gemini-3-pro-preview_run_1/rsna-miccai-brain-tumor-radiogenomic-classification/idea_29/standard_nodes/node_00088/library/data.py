import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Safe import for pydicom as it might not be explicitly listed but is available in environment
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config
from library.utils import get_logger, save_numpy_cache, load_numpy_cache

logger = get_logger("data_loader")


def natural_sort_key(s):
    """
    Sorts strings that contain numbers in a human-natural way.
    e.g. Image-2.dcm comes before Image-10.dcm
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom(path, target_size=Config.IMG_SIZE):
    """
    Reads a DICOM file and returns a numpy array.
    Tries pydicom first, then cv2.
    Resizes to target_size and normalizes to [0, 1].
    """
    img = None
    # Attempt 1: pydicom
    if pydicom:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
        except Exception:
            pass

    # Attempt 2: OpenCV
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Return zeros
    if img is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Normalize to [0, 1]
    img = img.astype(np.float32)
    if img.max() > 0:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img)

    # Resize if necessary
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return img


def get_brain_roi_indices(file_paths):
    """
    Determines the start and end indices of the brain tissue within a sorted list of files.
    Uses a stride-based scan to be efficient while robust.
    """
    if not file_paths:
        return 0, 0

    n_files = len(file_paths)
    if n_files < 5:
        return 0, n_files - 1

    first_brain = 0
    last_brain = n_files - 1

    # Heuristic: Scan with stride 2 to find brain start
    for i in range(0, n_files, 2):
        img = read_dicom(file_paths[i])
        if img.max() > 0.05:  # Threshold for "brain presence"
            first_brain = i
            break

    # Scan from end to find brain end
    for i in range(n_files - 1, -1, -2):
        if i <= first_brain:
            break
        img = read_dicom(file_paths[i])
        if img.max() > 0.05:
            last_brain = i
            break

    return first_brain, last_brain


def process_subject(row, input_dir):
    """
    Loads and processes a single subject into a 9-channel tensor.
    Channels 0-2: [FLAIR, T1wCE, T2w] at 40% depth
    Channels 3-5: [FLAIR, T1wCE, T2w] at 50% depth
    Channels 6-8: [FLAIR, T1wCE, T2w] at 60% depth
    """
    mods = Config.SELECTED_MODALITIES
    depths = Config.RELATIVE_DEPTHS  # [0.4, 0.5, 0.6]

    channels = []

    # Pre-calculate ROI for each modality to ensure independent alignment
    mod_info = {}
    for mod in mods:
        # Construct path relative to input dir
        rel_path = row[f"{mod.lower()}_path"]
        full_path = os.path.join(input_dir, rel_path)

        files = []
        if os.path.exists(full_path):
            files = glob.glob(os.path.join(full_path, "*.dcm"))
            files.sort(key=natural_sort_key)

        start, end = get_brain_roi_indices(files)
        roi_len = end - start + 1

        mod_info[mod] = {"files": files, "start": start, "roi_len": roi_len}

    # Build the stack: Iterate Depths (outer) then Modalities (inner)
    for depth_ratio in depths:
        for mod in mods:
            info = mod_info[mod]
            files = info["files"]

            if not files:
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            else:
                # Calculate index relative to ROI
                idx = info["start"] + int(info["roi_len"] * depth_ratio)
                idx = min(max(idx, 0), len(files) - 1)
                img = read_dicom(files[idx])

            channels.append(img)

    # Stack to (H, W, 9)
    stack = np.stack(channels, axis=-1)
    return stack


def load_data(split, load_cached_data=True):
    """
    Loads data for a specific split. Handles caching to .npy files.
    """
    # Define paths
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_img_path = Config.CACHE_TRAIN_IMAGES
        cache_lbl_path = Config.CACHE_TRAIN_LABELS
        cache_id_path = Config.CACHE_TRAIN_IDS
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        cache_img_path = Config.CACHE_VAL_IMAGES
        cache_lbl_path = Config.CACHE_VAL_LABELS
        cache_id_path = Config.CACHE_VAL_IDS
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        cache_img_path = Config.CACHE_TEST_IMAGES
        cache_lbl_path = None
        cache_id_path = Config.CACHE_TEST_IDS
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load metadata first to determine expected size
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    expected_size = len(df)
    logger.info(f"Expected {split} set size: {expected_size}")

    # Attempt to load from cache
    if load_cached_data:
        images = load_numpy_cache(cache_img_path)
        ids = load_numpy_cache(cache_id_path)
        if split != "test":
            labels = load_numpy_cache(cache_lbl_path)
        else:
            labels = None

        if images is not None and ids is not None:
            # Verify cache consistency (Cite debug_lesson_1)
            if len(ids) == expected_size:
                if split == "test" or labels is not None:
                    logger.info(f"Loaded {split} data from cache.")
                    return ids, images, labels
            else:
                logger.warning(
                    f"Cache size mismatch (Cache: {len(ids)}, Expected: {expected_size}). "
                    "Ignoring cache and reloading from scratch."
                )

    # Process from scratch
    logger.info(f"Processing {split} data from scratch...")

    ids_list = []
    images_list = []
    labels_list = []

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        img_stack = process_subject(row, Config.INPUT_DIR)  # (H, W, 9)

        ids_list.append(sid)
        images_list.append(img_stack)

        if split != "test":
            labels_list.append(row["MGMT_value"])

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{expected_size} subjects")

    # Convert to numpy
    ids_np = np.array(ids_list)
    images_np = np.array(images_list, dtype=np.float32)

    if split != "test":
        labels_np = np.array(labels_list, dtype=np.float32)
    else:
        labels_np = None

    # Strict Integrity Verification
    if len(ids_np) != expected_size:
        logger.warning(
            f"Data truncation detected! Expected {expected_size}, got {len(ids_np)}"
        )

    # Save to cache
    save_numpy_cache(ids_np, cache_id_path)
    save_numpy_cache(images_np, cache_img_path)
    if split != "test":
        save_numpy_cache(labels_np, cache_lbl_path)

    return ids_np, images_np, labels_np


def get_transforms(phase):
    """
    Returns Albumentations transforms for Spatially-Preserved Augmentation.
    """
    if phase == "train":
        return A.Compose(
            [
                # Spatial deformations that preserve relative anatomy
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Note: ShiftScaleRotate is excluded to preserve centroid alignment
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class MGMTDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, mode="val"):
        self.images = images  # (N, H, W, 9)
        self.labels = labels
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, 9)
        image = self.images[idx]

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Tensor (9, H, W)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Dummy label for test set
            return image, torch.tensor(0.0, dtype=torch.float32)


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    load_cached_data=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates a DataLoader for the specified split.
    """
    ids, images, labels = load_data(split, load_cached_data=load_cached_data)

    mode = "train" if split == "train" else "val"
    transform = get_transforms(mode)

    dataset = MGMTDataset(images, labels, transform=transform, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
