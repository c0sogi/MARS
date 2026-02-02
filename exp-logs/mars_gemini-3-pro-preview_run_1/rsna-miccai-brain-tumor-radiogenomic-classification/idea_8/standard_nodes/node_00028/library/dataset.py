import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import read_dicom_processed, get_middle_indices


def natural_sort_key(s):
    """
    Key for natural sorting of strings containing numbers (e.g., Image-1, Image-2, Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def process_subject(row, input_dir, img_size, slice_depth):
    """
    Loads, processes, and stacks MRI slices for a single subject.
    Returns a (H, W, C) float32 numpy array.
    """
    # Modalities: FLAIR, T1wCE, T2w
    channels = []

    for mod_col in Config.MODALITY_COLS:
        rel_path = row[mod_col]
        full_path = os.path.join(input_dir, rel_path)

        # List and sort files
        if os.path.exists(full_path):
            files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
            files.sort(key=natural_sort_key)
        else:
            files = []

        # Select middle slices
        indices = get_middle_indices(files, slice_depth)

        modality_slices = []
        for idx in indices:
            f_path = os.path.join(full_path, files[idx])
            # read_dicom_processed returns float32 (H, W)
            img = read_dicom_processed(f_path, img_size)
            modality_slices.append(img)

        # Handle missing slices (pad with zeros)
        while len(modality_slices) < slice_depth:
            modality_slices.append(np.zeros((img_size, img_size), dtype=np.float32))

        # Stack slices for this modality to compute stats: (Depth, H, W)
        mod_stack = np.array(modality_slices)

        # Independent Channel Min-Max Scaling
        min_val = np.min(mod_stack)
        max_val = np.max(mod_stack)

        if max_val - min_val > 1e-8:
            mod_stack = (mod_stack - min_val) / (max_val - min_val)
        else:
            mod_stack = np.zeros_like(mod_stack)

        # Append normalized slices to channel list
        for i in range(slice_depth):
            channels.append(mod_stack[i])

    # Stack all channels: (Total_Channels, H, W) -> Transpose to (H, W, Total_Channels)
    # Total channels = 3 modalities * 3 slices = 9
    combined = np.stack(channels, axis=-1)
    return combined


def load_or_generate_data(df, cache_prefix, load_cached_data):
    """
    Loads data from .npy cache if available, otherwise processes DICOMs and saves to cache.
    """
    # Cite debug_lesson_5: Bind Cache Identity to Data Generation Hyperparameters
    cache_suffix = f"_d{Config.SLICE_DEPTH}_s{Config.IMG_SIZE}"
    img_cache_path = os.path.join(
        Config.WORKING_DIR, f"{cache_prefix}_images{cache_suffix}.npy"
    )
    ids_cache_path = os.path.join(
        Config.WORKING_DIR, f"{cache_prefix}_ids{cache_suffix}.npy"
    )
    lbl_cache_path = os.path.join(
        Config.WORKING_DIR, f"{cache_prefix}_labels{cache_suffix}.npy"
    )

    has_labels = "MGMT_value" in df.columns

    # Try loading from cache
    if load_cached_data:
        if os.path.exists(img_cache_path) and os.path.exists(ids_cache_path):
            if has_labels and not os.path.exists(lbl_cache_path):
                pass  # Labels missing, regenerate
            else:
                print(
                    f"Loading cached {cache_prefix} data from {Config.WORKING_DIR}..."
                )
                images = np.load(img_cache_path)
                ids = np.load(ids_cache_path)

                # Cite debug_lesson_1: Verify Cache Consistency
                if len(images) != len(df):
                    print(
                        f"Cache length mismatch ({len(images)} vs {len(df)}). Regenerating..."
                    )
                else:
                    labels = np.load(lbl_cache_path) if has_labels else None
                    return images, ids, labels

    print(f"Processing and caching {cache_prefix} data (this may take a while)...")
    images_list = []
    ids_list = []
    labels_list = []

    for idx, row in df.iterrows():
        img = process_subject(
            row, Config.INPUT_DIR, Config.IMG_SIZE, Config.SLICE_DEPTH
        )
        images_list.append(img)
        ids_list.append(row["BraTS21ID"])
        if has_labels:
            labels_list.append(row["MGMT_value"])

    # Convert to pure numpy arrays (no pickling of objects)
    images = np.array(images_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=np.int64)
    labels = np.array(labels_list, dtype=np.float32) if labels_list else None

    # Save to cache
    np.save(img_cache_path, images)
    np.save(ids_cache_path, ids)
    if labels is not None:
        np.save(lbl_cache_path, labels)

    return images, ids, labels


class BraTSDataset(Dataset):
    def __init__(self, images, ids, labels=None, transform=None):
        self.images = images
        self.ids = ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image is (H, W, C)
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns (C, H, W) tensor
        else:
            # Manual conversion if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            id_val = self.ids[idx]
            return image, id_val


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(load_cached_data=True, batch_size=None):
    """
    Main interface to get data loaders.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Loading only {Config.DEBUG_SAMPLES} samples per split.")
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        test_df = test_df.head(Config.DEBUG_SAMPLES)

    # Process or Load Data
    train_imgs, train_ids, train_lbls = load_or_generate_data(
        train_df, "train", load_cached_data
    )
    val_imgs, val_ids, val_lbls = load_or_generate_data(val_df, "val", load_cached_data)
    test_imgs, test_ids, _ = load_or_generate_data(test_df, "test", load_cached_data)

    # Instantiate Datasets
    train_dataset = BraTSDataset(
        train_imgs, train_ids, train_lbls, transform=get_transforms("train")
    )
    val_dataset = BraTSDataset(
        val_imgs, val_ids, val_lbls, transform=get_transforms("val")
    )
    test_dataset = BraTSDataset(
        test_imgs, test_ids, labels=None, transform=get_transforms("test")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
