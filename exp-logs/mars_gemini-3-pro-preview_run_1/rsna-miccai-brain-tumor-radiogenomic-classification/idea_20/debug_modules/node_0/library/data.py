import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train' or 'valid'.
    """
    if phase == "train":
        return A.Compose(
            [
                # Conservative Augmentation: Geometric transforms only
                # We apply the same geometric transformation to all 9 channels simultaneously
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                # Note: No RandomBrightnessContrast or similar pixel-level augs
                # as they might distort the MRI intensity semantics preserved by min-max scaling.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def load_dicom_slice(path: str, img_size: int) -> np.ndarray:
    """
    Loads a single DICOM slice, resizes it, and applies min-max normalization.
    Returns a float32 array of shape (H, W).
    """
    img = None
    # Attempt 1: pydicom
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
    except Exception:
        pass

    # Attempt 2: cv2
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        # Fallback: return zero image
        return np.zeros((img_size, img_size), dtype=np.float32)

    # Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Convert to float32
    img = img.astype(np.float32)

    # Independent Channel Min-Max Scaling to [0, 1]
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def get_sorted_files(folder_path: str):
    """
    Returns a sorted list of DICOM files in a directory.
    Sorts numerically by the index in 'Image-X.dcm'.
    """
    if not os.path.exists(folder_path):
        return []

    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]

    def sort_key(f_name):
        # Extract number from Image-123.dcm
        match = re.search(r"Image-(\d+)", f_name)
        if match:
            return int(match.group(1))
        return f_name  # Fallback

    return sorted(files, key=sort_key)


def process_subject(row, input_dir: str, img_size: int, stride: int) -> np.ndarray:
    """
    Generates the 9-channel volumetric stack for a single subject.
    Channels 0-2: [FLAIR, T1wCE, T2w] at Depth M - stride
    Channels 3-5: [FLAIR, T1wCE, T2w] at Depth M
    Channels 6-8: [FLAIR, T1wCE, T2w] at Depth M + stride
    """
    modalities = ["flair", "t1wce", "t2w"]  # T1w excluded as per idea description

    stacked_volume = []

    # Pre-fetch file lists and calculate medians
    mod_files_map = {}
    mod_median_map = {}

    for mod in modalities:
        rel_path = row[f"{mod}_path"]
        full_path = os.path.join(input_dir, rel_path)
        files = get_sorted_files(full_path)
        mod_files_map[mod] = files
        mod_median_map[mod] = len(files) // 2

    offsets = [-stride, 0, stride]

    # Interleave: For each depth, add all modalities
    for offset in offsets:
        for mod in modalities:
            files = mod_files_map[mod]
            median_idx = mod_median_map[mod]
            target_idx = median_idx + offset

            # Handle edge cases (missing files or indices out of bounds)
            if len(files) == 0:
                img = np.zeros((img_size, img_size), dtype=np.float32)
            else:
                # Clamp index to valid range
                target_idx = max(0, min(target_idx, len(files) - 1))
                file_name = files[target_idx]
                file_path = os.path.join(input_dir, row[f"{mod}_path"], file_name)
                img = load_dicom_slice(file_path, img_size)

            stacked_volume.append(img)

    # Stack along channel axis (last axis)
    # List of (224, 224) -> (224, 224, 9)
    volume = np.stack(stacked_volume, axis=-1)
    return volume


def process_and_cache_data(metadata_path: str, cache_prefix: str, load_cache: bool):
    """
    Loads metadata, processes images (or loads from cache), and returns arrays.
    """
    cache_img_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    cache_lbl_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try Loading Cache
    if load_cache and os.path.exists(cache_img_path) and os.path.exists(cache_ids_path):
        print(f"Loading cached data for {cache_prefix}...")
        images = np.load(cache_img_path)
        ids = np.load(cache_ids_path)
        if os.path.exists(cache_lbl_path):
            labels = np.load(cache_lbl_path)
        else:
            labels = None
        return images, labels, ids

    # 2. Process from Scratch
    print(f"Processing data for {cache_prefix} from scratch...")
    df = pd.read_csv(metadata_path)

    images_list = []
    labels_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process Volume
        vol = process_subject(row, Config.INPUT_DIR, Config.IMG_SIZE, Config.STRIDE)
        images_list.append(vol)

        # ID
        ids_list.append(row["BraTS21ID"])

        # Label (if exists)
        if "MGMT_value" in row:
            labels_list.append(row["MGMT_value"])
        else:
            labels_list.append(-1.0)  # Dummy placeholder for test set

    # Convert to numpy
    images = np.array(images_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=np.int64)

    # Handle labels
    if labels_list and labels_list[0] != -1.0:
        labels = np.array(labels_list, dtype=np.float32)
        np.save(cache_lbl_path, labels)
    else:
        labels = None

    # Save to Cache
    np.save(cache_img_path, images)
    np.save(cache_ids_path, ids)

    return images, labels, ids


class WIVSDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray = None,
        ids: np.ndarray = None,
        transform=None,
    ):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, 9) float32
        image = self.images[idx]

        if self.transform:
            # Albumentations expects 'image'
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Return dummy label for test set
            return image, torch.tensor(0.0, dtype=torch.float32)


def get_datasets(load_cached_data: bool = True):
    """
    Main entry point to get the datasets.
    Handles caching logic internally.
    """
    # Ensure cache dir exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Train
    train_imgs, train_lbls, _ = process_and_cache_data(
        Config.TRAIN_METADATA, "train", load_cached_data
    )
    train_dataset = WIVSDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    # 2. Validation
    val_imgs, val_lbls, _ = process_and_cache_data(
        Config.VAL_METADATA, "val", load_cached_data
    )
    val_dataset = WIVSDataset(val_imgs, val_lbls, transform=get_transforms("valid"))

    # 3. Test
    test_imgs, _, test_ids = process_and_cache_data(
        Config.TEST_METADATA, "test", load_cached_data
    )
    test_dataset = WIVSDataset(
        test_imgs, labels=None, ids=test_ids, transform=get_transforms("valid")
    )

    return train_dataset, val_dataset, test_dataset
