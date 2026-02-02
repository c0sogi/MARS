import os
import re
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config
from library.utils import print_metric


def get_transforms(phase: str):
    """
    Returns the albumentations transformations for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def read_dicom_slice(path: str, img_size: int):
    """
    Reads a DICOM file, resizes it, and applies Min-Max normalization.
    Returns a float32 array of shape (img_size, img_size).
    """
    if not os.path.exists(path):
        return np.zeros((img_size, img_size), dtype=np.float32)

    img = None
    # Attempt to use pydicom if available
    if pydicom:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
        except Exception:
            pass

    # Fallback to OpenCV if pydicom failed or is unavailable
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        return np.zeros((img_size, img_size), dtype=np.float32)

    # Resize to target dimensions
    try:
        img = cv2.resize(img, (img_size, img_size))
    except Exception:
        return np.zeros((img_size, img_size), dtype=np.float32)

    # Independent Channel Min-Max Scaling to [0, 1]
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def get_file_list(folder_path):
    """
    Returns a sorted list of file numbers and a map of index to full path.
    """
    if not os.path.exists(folder_path):
        return [], {}

    files = os.listdir(folder_path)
    file_map = {}
    indices = []

    for f in files:
        if f.endswith(".dcm"):
            # Extract number from Image-X.dcm
            match = re.search(r"Image-(\d+)\.dcm", f)
            if match:
                idx = int(match.group(1))
                indices.append(idx)
                file_map[idx] = os.path.join(folder_path, f)

    indices.sort()
    return indices, file_map


def load_subject_slabs(row, input_dir, img_size, modalities):
    """
    Generates a single 3-channel image for a single subject using
    the median slice of each modality.

    Returns:
        images: (1, img_size, img_size, 3) float32 array
    """
    # 1. Identify Median Indices for each modality independently
    medians = {}
    file_maps = {}

    for mod in modalities:
        # Construct path relative to input dir
        rel_path = row[f"{mod.lower()}_path"]
        full_path = os.path.join(input_dir, rel_path)

        indices, f_map = get_file_list(full_path)

        if indices:
            medians[mod] = indices[len(indices) // 2]
            file_maps[mod] = f_map
        else:
            medians[mod] = 0
            file_maps[mod] = {}

    # 2. Extract Middle Slice (Geometric Heuristic)
    channels = []

    for mod in modalities:
        center = medians[mod]
        f_path = file_maps[mod].get(center, None)

        if f_path:
            img = read_dicom_slice(f_path, img_size)
        else:
            img = np.zeros((img_size, img_size), dtype=np.float32)
        channels.append(img)

    # Stack channels: (img_size, img_size, 3)
    # Cite solution_lesson_node_00009: Avoid naively stacking depth. Use standard RGB-like input.
    img_3ch = np.stack(channels, axis=-1)

    # Return as a list of 1 to maintain compatibility with downstream loops
    return np.array([img_3ch], dtype=np.float32)


class WIISDataset(Dataset):
    """
    Dataset class for the Weight-Inflated Independent-Slab Network.
    Wraps pre-processed numpy arrays and applies volumetric augmentations.
    """

    def __init__(self, images, labels=None, ids=None, transforms=None, is_test=False):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Apply transforms (Geometric transforms applied to all 9 channels simultaneously)
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Fallback to manual tensor conversion if no transforms provided
            img = torch.from_numpy(img.transpose(2, 0, 1))

        if self.is_test:
            # Return image and Subject ID for consensus aggregation
            subject_id = self.ids[idx] if self.ids is not None else 0
            return img, subject_id
        else:
            # Return image and Label
            label = self.labels[idx]
            return img, torch.tensor(label, dtype=torch.float32)


def prepare_data(load_cached_data=True):
    """
    Prepares the dataset: loads metadata, processes images into slabs (or loads cache),
    and returns datasets for train, val, and test.

    Implements the caching mechanism using .npy files in the working directory.
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # =========================================================
    # 1. TRAIN DATA
    # =========================================================
    train_cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_IMAGES)
        and os.path.exists(Config.CACHE_TRAIN_LABELS)
        and os.path.exists(Config.CACHE_TRAIN_IDS)
    )

    if load_cached_data and train_cache_exists:
        print("Loading cached training data...")
        train_images = np.load(Config.CACHE_TRAIN_IMAGES)
        train_labels = np.load(Config.CACHE_TRAIN_LABELS)
        train_ids = np.load(Config.CACHE_TRAIN_IDS)
    else:
        print("Processing training data from scratch...")
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        if Config.DEBUG:
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)

        train_imgs_list = []
        train_lbls_list = []
        train_ids_list = []

        for _, row in df_train.iterrows():
            sid = row["BraTS21ID"]
            label = row["MGMT_value"]

            slabs = load_subject_slabs(
                row,
                Config.INPUT_DIR,
                Config.IMG_SIZE,
                Config.SELECTED_MODALITIES,
            )

            # Add the single representative slice
            # Cite solution_lesson_node_00058: Do not treat correlated views as independent instances.
            train_imgs_list.append(slabs[0])
            train_lbls_list.append(label)
            train_ids_list.append(sid)

        train_images = np.array(train_imgs_list, dtype=np.float32)
        train_labels = np.array(train_lbls_list, dtype=np.float32)
        train_ids = np.array(train_ids_list, dtype=np.int64)

        # Save cache
        np.save(Config.CACHE_TRAIN_IMAGES, train_images)
        np.save(Config.CACHE_TRAIN_LABELS, train_labels)
        np.save(Config.CACHE_TRAIN_IDS, train_ids)

    # =========================================================
    # 2. VALIDATION DATA
    # =========================================================
    val_cache_exists = (
        os.path.exists(Config.CACHE_VAL_IMAGES)
        and os.path.exists(Config.CACHE_VAL_LABELS)
        and os.path.exists(Config.CACHE_VAL_IDS)
    )

    if load_cached_data and val_cache_exists:
        print("Loading cached validation data...")
        val_images = np.load(Config.CACHE_VAL_IMAGES)
        val_labels = np.load(Config.CACHE_VAL_LABELS)
        val_ids = np.load(Config.CACHE_VAL_IDS)
    else:
        print("Processing validation data from scratch...")
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        if Config.DEBUG:
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)

        val_imgs_list = []
        val_lbls_list = []
        val_ids_list = []

        for _, row in df_val.iterrows():
            sid = row["BraTS21ID"]
            label = row["MGMT_value"]

            slabs = load_subject_slabs(
                row,
                Config.INPUT_DIR,
                Config.IMG_SIZE,
                Config.SELECTED_MODALITIES,
            )

            val_imgs_list.append(slabs[0])
            val_lbls_list.append(label)
            val_ids_list.append(sid)

        val_images = np.array(val_imgs_list, dtype=np.float32)
        val_labels = np.array(val_lbls_list, dtype=np.float32)
        val_ids = np.array(val_ids_list, dtype=np.int64)

        np.save(Config.CACHE_VAL_IMAGES, val_images)
        np.save(Config.CACHE_VAL_LABELS, val_labels)
        np.save(Config.CACHE_VAL_IDS, val_ids)

    # =========================================================
    # 3. TEST DATA
    # =========================================================
    test_cache_exists = os.path.exists(Config.CACHE_TEST_IMAGES) and os.path.exists(
        Config.CACHE_TEST_IDS
    )

    if load_cached_data and test_cache_exists:
        print("Loading cached test data...")
        test_images = np.load(Config.CACHE_TEST_IMAGES)
        test_ids = np.load(Config.CACHE_TEST_IDS)
    else:
        print("Processing test data from scratch...")
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        if Config.DEBUG:
            df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

        test_imgs_list = []
        test_ids_list = []

        for _, row in df_test.iterrows():
            sid = row["BraTS21ID"]

            slabs = load_subject_slabs(
                row,
                Config.INPUT_DIR,
                Config.IMG_SIZE,
                Config.SELECTED_MODALITIES,
            )

            test_imgs_list.append(slabs[0])
            test_ids_list.append(sid)

        test_images = np.array(test_imgs_list, dtype=np.float32)
        test_ids = np.array(test_ids_list, dtype=np.int64)

        np.save(Config.CACHE_TEST_IMAGES, test_images)
        np.save(Config.CACHE_TEST_IDS, test_ids)

    # =========================================================
    # 4. DATASET CREATION
    # =========================================================
    train_dataset = WIISDataset(
        train_images,
        train_labels,
        train_ids,
        transforms=get_transforms("train"),
        is_test=False,
    )

    val_dataset = WIISDataset(
        val_images, val_labels, val_ids, transforms=get_transforms("val"), is_test=False
    )

    test_dataset = WIISDataset(
        test_images,
        labels=None,
        ids=test_ids,
        transforms=get_transforms("test"),
        is_test=True,
    )

    print(f"Data preparation complete.")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
