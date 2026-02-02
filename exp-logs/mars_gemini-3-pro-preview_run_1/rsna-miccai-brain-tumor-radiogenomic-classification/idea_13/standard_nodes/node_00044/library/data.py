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

from library.config import Config
from library.utils import set_seed

# ==========================================
# Helper Functions
# ==========================================


def natural_key(string_):
    """
    Key for natural sorting of filenames (e.g., Image-1, Image-2, Image-10).
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def get_sorted_file_list(folder_path):
    """
    Returns a naturally sorted list of DICOM files in a directory.
    """
    if not os.path.exists(folder_path):
        return []
    files = glob.glob(os.path.join(folder_path, "*.dcm"))
    files.sort(key=lambda x: natural_key(os.path.basename(x)))
    return files


def select_indices(file_list, stride=Config.STRIDE):
    """
    Selects the median index.
    Cite Lesson 00015: Prefer deterministic geometric heuristics (such as the median slice).
    Cite Lesson 00018: A simple model focusing on the single most informative instance often outperforms complex aggregation.
    """
    n = len(file_list)
    if n == 0:
        return [0]

    mid = n // 2
    return [mid]


def load_dicom_as_float(path, target_size=Config.IMG_SIZE):
    """
    Reads a DICOM file, resizes it, and normalizes to [0, 1] float32.
    Attempts pydicom first, falls back to cv2.
    """
    img = None

    # Attempt 1: pydicom
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
    except (ImportError, Exception):
        pass

    # Attempt 2: cv2
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Empty image
    if img is None:
        img = np.zeros((target_size, target_size), dtype=np.float32)

    # Resize
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Convert to float32
    img = img.astype(np.float32)

    # Min-Max Normalization to [0, 1]
    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)  # Avoid division by zero

    return img


def process_subset(metadata_df, subset_name, input_dir="./input"):
    """
    Processes a subset of data (train/val/test) into numpy arrays.
    Applies Wide-Field Stratified Instance Learning logic.
    """
    images = []
    labels = []
    ids = []

    print(f"Processing {subset_name} data: {len(metadata_df)} subjects...")

    # Limit for debugging if configured
    if Config.MAX_SAMPLES is not None:
        metadata_df = metadata_df.head(Config.MAX_SAMPLES)
        print(f"DEBUG: Limited to {Config.MAX_SAMPLES} subjects.")

    for _, row in metadata_df.iterrows():
        bra_id = row["BraTS21ID"]
        # Handle label existence (Test set has no label)
        label = row["MGMT_value"] if "MGMT_value" in row else -1

        # Paths
        flair_dir = os.path.join(input_dir, row["flair_path"])
        t1wce_dir = os.path.join(input_dir, row["t1wce_path"])
        t2w_dir = os.path.join(input_dir, row["t2w_path"])

        # Get sorted file lists
        flair_files = get_sorted_file_list(flair_dir)
        t1wce_files = get_sorted_file_list(t1wce_dir)
        t2w_files = get_sorted_file_list(t2w_dir)

        # Determine indices independently
        flair_idxs = select_indices(flair_files)
        t1wce_idxs = select_indices(t1wce_files)
        t2w_idxs = select_indices(t2w_files)

        # If any modality is missing files, skip subject or handle gracefully
        # (Here we proceed, load_dicom handles empty paths by returning zeros if list empty)

        # Process selected instances (Single Middle Slice)
        # Cite Lesson 00015: Prefer deterministic geometric heuristics.
        for i in range(1):
            # Load slices
            f_path = flair_files[flair_idxs[i]] if flair_files else ""
            c_path = t1wce_files[t1wce_idxs[i]] if t1wce_files else ""
            t_path = t2w_files[t2w_idxs[i]] if t2w_files else ""

            img_f = load_dicom_as_float(f_path)
            img_c = load_dicom_as_float(c_path)
            img_t = load_dicom_as_float(t_path)

            # Stack channels: (H, W, 3)
            # We stack along the last axis for Albumentations compatibility
            img_stack = np.dstack((img_f, img_c, img_t))

            images.append(img_stack)
            labels.append(label)
            ids.append(bra_id)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(ids, dtype=np.int64),
    )


def get_data(load_cached_data=True):
    """
    Main function to load data. Checks cache first, otherwise processes from scratch.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Training Data
    # ---------------------------------------------------------
    train_exists = (
        os.path.exists(Config.CACHE_TRAIN_IMAGES)
        and os.path.exists(Config.CACHE_TRAIN_LABELS)
        and os.path.exists(Config.CACHE_TRAIN_IDS)
    )

    if load_cached_data and train_exists:
        print("Loading cached Training data...")
        train_images = np.load(Config.CACHE_TRAIN_IMAGES)
        train_labels = np.load(Config.CACHE_TRAIN_LABELS)
        train_ids = np.load(Config.CACHE_TRAIN_IDS)
    else:
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        train_images, train_labels, train_ids = process_subset(df_train, "Train")
        # Cache
        np.save(Config.CACHE_TRAIN_IMAGES, train_images)
        np.save(Config.CACHE_TRAIN_LABELS, train_labels)
        np.save(Config.CACHE_TRAIN_IDS, train_ids)
        print("Training data cached.")

    # ---------------------------------------------------------
    # 2. Validation Data
    # ---------------------------------------------------------
    val_exists = (
        os.path.exists(Config.CACHE_VAL_IMAGES)
        and os.path.exists(Config.CACHE_VAL_LABELS)
        and os.path.exists(Config.CACHE_VAL_IDS)
    )

    if load_cached_data and val_exists:
        print("Loading cached Validation data...")
        val_images = np.load(Config.CACHE_VAL_IMAGES)
        val_labels = np.load(Config.CACHE_VAL_LABELS)
        val_ids = np.load(Config.CACHE_VAL_IDS)
    else:
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        val_images, val_labels, val_ids = process_subset(df_val, "Validation")
        # Cache
        np.save(Config.CACHE_VAL_IMAGES, val_images)
        np.save(Config.CACHE_VAL_LABELS, val_labels)
        np.save(Config.CACHE_VAL_IDS, val_ids)
        print("Validation data cached.")

    # ---------------------------------------------------------
    # 3. Test Data
    # ---------------------------------------------------------
    test_exists = os.path.exists(Config.CACHE_TEST_IMAGES) and os.path.exists(
        Config.CACHE_TEST_IDS
    )

    if load_cached_data and test_exists:
        print("Loading cached Test data...")
        test_images = np.load(Config.CACHE_TEST_IMAGES)
        test_ids = np.load(Config.CACHE_TEST_IDS)
    else:
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_images, _, test_ids = process_subset(df_test, "Test")
        # Cache
        np.save(Config.CACHE_TEST_IMAGES, test_images)
        np.save(Config.CACHE_TEST_IDS, test_ids)
        print("Test data cached.")

    return (
        (train_images, train_labels, train_ids),
        (val_images, val_labels, val_ids),
        (test_images, test_ids),
    )


# ==========================================
# Dataset Class
# ==========================================


class WSILDataset(Dataset):
    """
    Dataset class for Wide-Field Stratified Instance Learning.
    Input images are expected to be (N, H, W, C) numpy arrays.
    """

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, 3)
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen with get_transforms)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


# ==========================================
# Transforms
# ==========================================


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for training or validation/test.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                # Images are already [0,1], ToTensorV2 converts HWC -> CHW
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# DataLoader Factory
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for Train, Val, and Test sets.
    """
    set_seed(Config.SEED)

    # Load data (from cache or process fresh)
    (train_imgs, train_lbls, _), (val_imgs, val_lbls, _), (test_imgs, test_ids) = (
        get_data(load_cached_data=load_cached_data)
    )

    # Create Datasets
    train_dataset = WSILDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    val_dataset = WSILDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    test_dataset = WSILDataset(test_imgs, labels=None, transform=get_transforms("test"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, test_ids
