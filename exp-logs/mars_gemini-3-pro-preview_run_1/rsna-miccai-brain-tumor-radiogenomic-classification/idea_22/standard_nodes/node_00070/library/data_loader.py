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
from sklearn.model_selection import StratifiedKFold
from library import config, utils

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm before Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom(path):
    """
    Reads a DICOM file and returns a numpy array.
    Tries pydicom first (if installed), then cv2.
    """
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
            return img
        except Exception:
            pass

    # Fallback to OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def load_volume(directory_path):
    """
    Loads all DICOM images from a directory into a 3D numpy array (Depth, Height, Width).
    """
    if not os.path.exists(directory_path):
        return np.zeros((0, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    files = glob.glob(os.path.join(directory_path, "*.dcm"))
    files.sort(key=natural_sort_key)

    images = []
    for f in files:
        img = read_dicom(f)
        if img is not None:
            images.append(img)

    if not images:
        return np.zeros((0, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    return np.array(images)


def get_centroid_index(volume):
    """
    Calculates the Z-axis Center of Mass of the brain tissue (pixels > 0).
    Returns the integer index.
    """
    if volume.shape[0] == 0:
        return 0

    # Threshold to find brain tissue
    # We use a simple threshold > 0.
    # To speed up, we can check max of each slice or sum of each slice
    slice_sums = np.sum(volume > 0, axis=(1, 2))
    indices = np.nonzero(slice_sums)[0]

    if len(indices) == 0:
        return volume.shape[0] // 2

    # Geometric centroid of the indices
    centroid = int(np.mean(indices))
    return centroid


def normalize_slice(img):
    """
    Min-Max normalization to [0, 1].
    """
    if img is None:
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def process_subject(row, input_dir):
    """
    Process a single subject:
    1. Load volumes for FLAIR, T1wCE, T2w.
    2. Compute centroids.
    3. Extract 3 slices per modality (z-d, z, z+d).
    4. Resize and Normalize.
    5. Stack into (H, W, 9).
    """
    # Channel structure:
    # 0-2: [FLAIR, T1wCE, T2w] at z - delta
    # 3-5: [FLAIR, T1wCE, T2w] at z
    # 6-8: [FLAIR, T1wCE, T2w] at z + delta

    modalities = config.MODALITIES  # ["FLAIR", "T1wCE", "T2w"]
    stride = config.SLICE_STRIDE

    # Store processed slices: list of list [modality_idx][depth_idx]
    # We want final order:
    # D0: M0, M1, M2
    # D1: M0, M1, M2
    # D2: M0, M1, M2

    # To achieve this, we collect slices and then stack properly

    # Container for the 3 channels
    channels = [None] * 3

    for m_idx, mod in enumerate(modalities):
        # Construct path
        rel_path = row[f"{mod.lower()}_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Load Volume
        vol = load_volume(full_path)

        # Compute Centroid
        if vol.shape[0] > 0:
            z = get_centroid_index(vol)
        else:
            z = 0

        # Extract Center Slice Only
        if vol.shape[0] > 0:
            slc = vol[z]
        else:
            slc = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

        # Resize
        if slc.shape[0] != config.IMG_SIZE or slc.shape[1] != config.IMG_SIZE:
            slc = cv2.resize(
                slc,
                (config.IMG_SIZE, config.IMG_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

        # Normalize
        slc = normalize_slice(slc)

        # Assign to correct channel position (0: FLAIR, 1: T1wCE, 2: T2w)
        channels[m_idx] = slc

    # Stack
    # Shape: (H, W, 9)
    img_stack = np.stack(channels, axis=-1)
    return img_stack


def load_and_cache_data(metadata_path, split_name, load_cached_data=True):
    """
    Loads data, processing it if not cached.
    Returns X (images) and y (targets/ids).
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_img_path = os.path.join(cache_dir, f"cached_{split_name}_images.npy")
    cache_lbl_path = os.path.join(cache_dir, f"cached_{split_name}_labels.npy")
    cache_ids_path = os.path.join(cache_dir, f"cached_{split_name}_ids.npy")

    # 1. Try Loading Cache
    if load_cached_data:
        if os.path.exists(cache_img_path) and os.path.exists(cache_ids_path):
            print(f"Loading cached {split_name} data from {cache_dir}...")
            X = np.load(cache_img_path)
            ids = np.load(cache_ids_path)

            # Load labels if they exist (test set might not have them in cache logic if we didn't save them,
            # but test metadata doesn't have labels column usually. We handle this.)
            if os.path.exists(cache_lbl_path):
                y = np.load(cache_lbl_path)
            else:
                y = None
            return X, y, ids

    # 2. Process from Scratch
    print(f"Processing {split_name} data from scratch...")
    df = pd.read_csv(metadata_path)

    # Debugging/Testing: Limit size if needed (not applied here for full run)
    # df = df.head(10)

    X_list = []
    y_list = []
    ids_list = []

    has_labels = "MGMT_value" in df.columns

    for idx, row in df.iterrows():
        # Process Image
        img_stack = process_subject(row, config.INPUT_DIR)
        X_list.append(img_stack)

        # Store ID
        ids_list.append(row["BraTS21ID"])

        # Store Label
        if has_labels:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    if has_labels:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to Cache
    np.save(cache_img_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_lbl_path, y)

    print(f"Saved {split_name} data to cache. Shape: {X.shape}")

    return X, y, ids


class VolumetricDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, 9)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor if no transforms provided (fallback)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    AC-WIV Strategy: Elastic, Grid, Rotate. NO Translation/Scaling.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.HorizontalFlip(p=0.5),
                # Normalization is done in preprocessing (MinMax per channel)
                # Just convert to tensor
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for the specified fold using StratifiedKFold.
    """
    # Load Training Data (which includes Val split implicitly via CV)
    # We combine train_metadata and val_metadata to perform our own CV split
    # or we can respect the provided split.
    # The prompt says: "5-Fold Cross-Validation grouped by Subject ID".
    # Since we have train_metadata and val_metadata separately, we should merge them
    # to perform a proper 5-fold CV on the whole development set.

    # Load Train Metadata
    X_train_raw, y_train_raw, ids_train = load_and_cache_data(
        config.TRAIN_METADATA_PATH, "train", load_cached_data
    )

    # Load Val Metadata
    X_val_raw, y_val_raw, ids_val = load_and_cache_data(
        config.VAL_METADATA_PATH, "val", load_cached_data
    )

    # Merge
    X_full = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_full = np.concatenate([y_train_raw, y_val_raw], axis=0)
    ids_full = np.concatenate([ids_train, ids_val], axis=0)

    # Create Folds
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the correct fold
    fold_generator = skf.split(X_full, y_full)
    train_idx, val_idx = next(x for i, x in enumerate(fold_generator) if i == fold_idx)

    # Split Data
    X_train_fold = X_full[train_idx]
    y_train_fold = y_full[train_idx]

    X_val_fold = X_full[val_idx]
    y_val_fold = y_full[val_idx]

    # Create Datasets
    train_dataset = VolumetricDataset(
        X_train_fold, y_train_fold, transforms=get_transforms("train")
    )

    val_dataset = VolumetricDataset(
        X_val_fold, y_val_fold, transforms=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    X_test, _, ids_test = load_and_cache_data(
        config.TEST_METADATA_PATH, "test", load_cached_data
    )

    test_dataset = VolumetricDataset(
        X_test, labels=None, transforms=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids_test
