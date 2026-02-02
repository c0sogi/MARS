import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from library import config
from library import utils

# ==========================================
# Augmentation Factory
# ==========================================


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for training or validation/test.

    Args:
        phase (str): 'train' or 'valid'.
    """
    if phase == "train":
        return A.Compose(
            [
                # Spatial transformations applied to all channels simultaneously
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                # Normalize is not needed here as we manually normalize to [0, 1] during loading
                # and EfficientNet usually expects specific mean/std, but our custom init
                # handles the distribution shift. We stick to [0,1] range.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# Data Processing Logic (WIVE Strategy)
# ==========================================


def _get_sorted_image_files(folder_path):
    """
    Returns a sorted list of DICOM files in a folder.
    Sorts by the integer instance number in the filename (e.g., Image-10.dcm).
    """
    if not os.path.exists(folder_path):
        return []

    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]

    # Extract number from filename for sorting: "Image-123.dcm" -> 123
    def extract_number(fname):
        match = re.search(r"(\d+)", fname)
        return int(match.group(1)) if match else 0

    files.sort(key=extract_number)
    return [os.path.join(folder_path, f) for f in files]


def _process_subject(row, input_dir, stride, img_size):
    """
    Extracts the 9-channel volumetric slab for a single subject.

    Channels mapping:
    0-2: FLAIR, T1wCE, T2w at Depth A (Median - Stride)
    3-5: FLAIR, T1wCE, T2w at Depth B (Median)
    6-8: FLAIR, T1wCE, T2w at Depth C (Median + Stride)
    """
    # Modalities to process
    modalities = config.MODALITIES  # ["FLAIR", "T1wCE", "T2w"]

    # Store slices for each modality at each depth
    # Structure: slices_by_modality[modality_idx] = [slice_A, slice_B, slice_C]
    slices_by_modality = []

    for mod in modalities:
        # Construct path: input_dir + relative_path_from_metadata
        # Metadata contains paths like 'train/00000/FLAIR'
        # row key is e.g., 'flair_path' (lowercase based on metadata generation script)
        col_name = f"{mod.lower()}_path"
        rel_path = row[col_name]
        full_path = os.path.join(input_dir, rel_path)

        files = _get_sorted_image_files(full_path)
        num_files = len(files)

        if num_files == 0:
            # Handle missing modality by creating zero arrays
            zero_img = np.zeros((img_size, img_size), dtype=np.float32)
            slices_by_modality.append([zero_img, zero_img, zero_img])
            continue

        # Determine indices
        median_idx = num_files // 2
        indices = [median_idx - stride, median_idx, median_idx + stride]

        # Clamp indices
        indices = [max(0, min(idx, num_files - 1)) for idx in indices]

        # Load and normalize slices
        mod_slices = []
        for idx in indices:
            img = utils.load_dicom_as_array(files[idx], size=img_size)
            img = utils.min_max_scale(img)
            mod_slices.append(img)

        slices_by_modality.append(mod_slices)

    # Stack into (9, H, W) tensor
    # Order:
    # Depth A (idx 0): FLAIR, T1wCE, T2w
    # Depth B (idx 1): FLAIR, T1wCE, T2w
    # Depth C (idx 2): FLAIR, T1wCE, T2w

    channels = []
    # Iterate over depths (0, 1, 2)
    for depth_idx in range(3):
        # Iterate over modalities (FLAIR, T1wCE, T2w)
        for mod_idx in range(len(modalities)):
            channels.append(slices_by_modality[mod_idx][depth_idx])

    # Stack along channel dimension (axis 0)
    volumetric_slab = np.stack(channels, axis=0)  # Shape: (9, 224, 224)

    return volumetric_slab.astype(np.float32)


def _generate_dataset_cache(metadata_df, save_prefix, load_cached_data=True):
    """
    Generates or loads cached numpy arrays for the dataset.
    """
    cache_dir = config.CACHE_DIR
    images_path = os.path.join(cache_dir, f"{save_prefix}_images.npy")
    ids_path = os.path.join(cache_dir, f"{save_prefix}_ids.npy")
    labels_path = os.path.join(cache_dir, f"{save_prefix}_labels.npy")

    has_labels = "MGMT_value" in metadata_df.columns

    # Check if cache exists
    if load_cached_data and os.path.exists(images_path) and os.path.exists(ids_path):
        if has_labels and not os.path.exists(labels_path):
            pass  # Labels missing, recompute
        else:
            print(f"Loading cached data from {cache_dir} ({save_prefix})...")
            images = np.load(images_path)
            ids = np.load(ids_path)
            labels = np.load(labels_path) if has_labels else None
            return images, labels, ids

    print(f"Processing data for {save_prefix} (Cache miss or force reload)...")

    images_list = []
    ids_list = []
    labels_list = []

    # Iterate through metadata
    # Use tqdm for progress if not silent, but prompt asks to minimize prints.
    # We will print a simple start/end message.

    for _, row in metadata_df.iterrows():
        # Process image
        img_tensor = _process_subject(
            row, config.INPUT_DIR, config.STRIDE, config.IMG_SIZE
        )
        images_list.append(img_tensor)

        # Store ID
        ids_list.append(row["BraTS21ID"])

        # Store Label
        if has_labels:
            labels_list.append(row["MGMT_value"])

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=np.int64)
    labels = np.array(labels_list, dtype=np.float32) if has_labels else None

    # Save to cache
    np.save(images_path, images)
    np.save(ids_path, ids)
    if labels is not None:
        np.save(labels_path, labels)

    print(f"Data processed and cached to {cache_dir}.")
    return images, labels, ids


# ==========================================
# Dataset Class
# ==========================================


class WIVEDataset(Dataset):
    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 9, H, W)
            labels (np.ndarray): Shape (N,)
            ids (np.ndarray): Shape (N,)
            transform (albumentations.Compose): Augmentations
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image: (9, H, W)
        img = self.images[idx]

        # Albumentations expects (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]  # Returns Tensor (C, H, W) via ToTensorV2
        else:
            # Fallback if no transform provided (shouldn't happen in this pipeline)
            img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        # If no labels (Test set), return ID for submission mapping
        if self.ids is not None:
            return img, self.ids[idx]

        return img


# ==========================================
# Public API
# ==========================================


def get_train_val_datasets(load_cached_data=True, debug=False):
    """
    Loads training and validation datasets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.
        debug (bool): If True, subsamples the data for quick testing.

    Returns:
        train_dataset, val_dataset
    """
    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    if debug:
        df_train = df_train.head(config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(config.DEBUG_SAMPLE_SIZE)

    # Process/Load Data
    train_imgs, train_lbls, train_ids = _generate_dataset_cache(
        df_train, "train" if not debug else "debug_train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = _generate_dataset_cache(
        df_val, "val" if not debug else "debug_val", load_cached_data
    )

    # Create Datasets
    train_dataset = WIVEDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = WIVEDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("valid")
    )

    return train_dataset, val_dataset


def get_test_dataset(load_cached_data=True):
    """
    Loads the test dataset.

    Returns:
        test_dataset
    """
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Process/Load Data
    test_imgs, _, test_ids = _generate_dataset_cache(df_test, "test", load_cached_data)

    test_dataset = WIVEDataset(
        test_imgs, labels=None, ids=test_ids, transform=get_transforms("valid")
    )

    return test_dataset


def get_dataloaders(train_dataset, val_dataset, batch_size=None):
    """
    Creates DataLoaders for training and validation.
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader
