import os
import cv2
import pydicom
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import library.config as config
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_IMAGES_DIR,
    TEST_IMAGES_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    TARGET_COLS,
    SEED,
    DEBUG,
    DEBUG_SIZE,
    IN_CHANNELS,
)


# ==========================================
# Caching & Path Management
# ==========================================
def get_study_paths_map(metadata_df, images_dir, cache_name, load_cached_data=True):
    """
    Generates or loads a dictionary mapping StudyInstanceUID to a sorted list of slice numbers.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing StudyInstanceUIDs.
        images_dir (str): Root directory containing study folders.
        cache_name (str): Name of the cache file (e.g., 'train_paths').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {StudyInstanceUID: [sorted_slice_numbers]}
    """
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}_cache.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading path cache from {cache_path}...")
            cache_df = pd.read_parquet(cache_path)
            # Group by StudyInstanceUID and collect slice_nums into lists
            # Assuming cache_df has columns: StudyInstanceUID, slice_num
            # We need to ensure slice_num is sorted.
            # Sorting the dataframe first ensures the lists are sorted.
            cache_df = cache_df.sort_values(["StudyInstanceUID", "slice_num"])
            path_map = (
                cache_df.groupby("StudyInstanceUID")["slice_num"].apply(list).to_dict()
            )
            return path_map
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Scanning directories in {images_dir}...")
    data_records = []

    study_uids = metadata_df["StudyInstanceUID"].unique()

    for uid in study_uids:
        study_path = os.path.join(images_dir, uid)
        if not os.path.exists(study_path):
            continue

        try:
            # List all files, filter for .dcm, parse integer slice number
            files = [f for f in os.listdir(study_path) if f.endswith(".dcm")]
            # Extract numbers: "10.dcm" -> 10
            # Filter out non-integer filenames just in case
            slices = []
            for f in files:
                try:
                    num = int(os.path.splitext(f)[0])
                    slices.append(num)
                except ValueError:
                    continue

            for s in slices:
                data_records.append({"StudyInstanceUID": uid, "slice_num": s})

        except OSError:
            continue

    # Create DataFrame
    cache_df = pd.DataFrame(data_records)

    # Sort for consistency
    cache_df = cache_df.sort_values(["StudyInstanceUID", "slice_num"])

    # Save to Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)
    print(f"Saved path cache to {cache_path}")

    # Convert to Dict
    path_map = cache_df.groupby("StudyInstanceUID")["slice_num"].apply(list).to_dict()
    return path_map


# ==========================================
# DICOM Processing
# ==========================================
def load_dicom_slice(path, img_size):
    """
    Reads a DICOM file, applies bone windowing, and resizes.

    Args:
        path (str): Path to the .dcm file.
        img_size (tuple): (height, width).

    Returns:
        np.ndarray: Processed image (H, W) normalized to 0-1.
    """
    try:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array

        # Rescale Intercept/Slope
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        slope = getattr(ds, "RescaleSlope", 1.0)
        pixel_array = pixel_array * slope + intercept

        # Bone Windowing
        # Center = 1000, Width = 2000 (Range: 0 to 2000 roughly)
        window_center = 1000
        window_width = 2000
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2

        img = np.clip(pixel_array, min_val, max_val)

        # Normalize to 0-1
        if max_val != min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img - min_val

    except Exception:
        # Fallback for corrupt or missing files: Black image
        img = np.zeros(img_size, dtype=np.float32)

    # Resize
    img = cv2.resize(img, (img_size[1], img_size[0]))

    return img.astype(np.float32)


# ==========================================
# Dataset Class
# ==========================================
class RSNADataset(Dataset):
    def __init__(
        self,
        metadata_df,
        path_map,
        images_dir,
        transform=None,
        seq_len=None,
        is_train=False,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata with StudyInstanceUID and labels.
            path_map (dict): Map of StudyUID -> List of sorted slice numbers.
            images_dir (str): Root directory of images.
            transform (A.Compose): Albumentations transforms.
            seq_len (int): Number of slices to sample.
            is_train (bool): Whether this is a training set (for augmentation logic).
        """
        self.metadata = metadata_df
        self.path_map = path_map
        self.images_dir = images_dir
        self.transform = transform
        self.seq_len = seq_len if seq_len is not None else config.SEQ_LEN
        self.is_train = is_train

        # Check if labels exist
        self.has_labels = all(col in metadata_df.columns for col in TARGET_COLS)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Get sorted slice numbers
        slice_nums = self.path_map.get(uid, [])
        num_slices = len(slice_nums)

        # 1. Uniform Sampling
        if num_slices == 0:
            # Fallback for empty study
            indices = np.zeros(self.seq_len, dtype=int)
            slice_nums = [0]  # Dummy
        else:
            # Sample uniformly across the Z-axis
            indices = np.linspace(0, num_slices - 1, self.seq_len).astype(int)

        # 2. Load Slices (2.5D Construction)
        # We need to load [z-1, z, z+1] for each sampled index z.
        # We will load them into a large buffer to apply geometric transforms efficiently.
        # Total channels = seq_len * 3

        loaded_images = []

        for i in indices:
            # Determine neighbors with clamping
            neighbors = [i - 1, i, i + 1]
            neighbors = [max(0, min(n, num_slices - 1)) for n in neighbors]

            # Load the 3 channels
            channels = []
            for n_idx in neighbors:
                if num_slices > 0:
                    s_num = slice_nums[n_idx]
                    path = os.path.join(self.images_dir, uid, f"{s_num}.dcm")
                    img = load_dicom_slice(path, IMG_SIZE)
                else:
                    img = np.zeros(IMG_SIZE, dtype=np.float32)
                channels.append(img)

            # Stack to (H, W, 3)
            img_25d = np.stack(channels, axis=-1)
            loaded_images.append(img_25d)

        # Stack all sequence frames along channel dimension for consistent augmentation
        # Shape: (H, W, seq_len * 3)
        volume_stack = np.concatenate(loaded_images, axis=-1)

        # 3. Augmentation
        if self.transform:
            # Albumentations expects (H, W, C)
            augmented = self.transform(image=volume_stack)["image"]
            # Result is tensor (C, H, W) if ToTensorV2 is used, or numpy (H, W, C)
            # We assume ToTensorV2 is NOT in the passed transform to handle splitting manually first,
            # OR we handle the tensor output.
            # Let's assume the transform returns a Tensor (C, H, W).
            volume_stack = augmented

        # 4. Reshape to (Seq_Len, 3, H, W)
        # If tensor: (Seq_Len * 3, H, W) -> (Seq_Len, 3, H, W)
        if isinstance(volume_stack, torch.Tensor):
            # Reshape
            c, h, w = volume_stack.shape
            volume_stack = volume_stack.view(self.seq_len, 3, h, w)
        else:
            # If numpy: (H, W, Seq_Len * 3) -> (Seq_Len, H, W, 3) -> (Seq_Len, 3, H, W)
            h, w, c = volume_stack.shape
            volume_stack = volume_stack.reshape(h, w, self.seq_len, 3)
            volume_stack = np.transpose(volume_stack, (2, 3, 0, 1))
            volume_stack = torch.from_numpy(volume_stack)

        # 5. Get Labels
        if self.has_labels:
            labels = row[TARGET_COLS].values.astype(np.float32)
            labels = torch.tensor(labels)
        else:
            labels = torch.zeros(len(TARGET_COLS), dtype=np.float32)

        return volume_stack, labels


# ==========================================
# Data Loaders
# ==========================================
def get_dataloaders(
    batch_size=BATCH_SIZE,
    load_cached_data=True,
    debug=DEBUG,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached path maps.
        debug (bool): If True, subsamples the dataset.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    if debug:
        train_df = train_df.iloc[:DEBUG_SIZE]
        val_df = val_df.iloc[:DEBUG_SIZE]
        test_df = test_df.iloc[:DEBUG_SIZE]
        print(f"DEBUG Mode: Subsampled datasets to {DEBUG_SIZE} samples.")

    # 2. Prepare Path Maps (Cache)
    # Train and Val share the TRAIN_IMAGES_DIR
    train_path_map = get_study_paths_map(
        pd.concat([train_df, val_df]), TRAIN_IMAGES_DIR, "train_paths", load_cached_data
    )

    test_path_map = get_study_paths_map(
        test_df, TEST_IMAGES_DIR, "test_paths", load_cached_data
    )

    # 3. Define Transforms
    # We apply geometric transforms to the stacked volume.
    # Note: We do NOT use ToTensorV2 here because we need to reshape numpy array first usually,
    # but RSNADataset handles numpy->tensor conversion manually to ensure correct reshaping.
    # However, using ToTensorV2 is convenient for channel-first permutation.

    train_transform = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    # 4. Create Datasets
    train_dataset = RSNADataset(
        train_df,
        train_path_map,
        TRAIN_IMAGES_DIR,
        transform=train_transform,
        is_train=True,
    )

    val_dataset = RSNADataset(
        val_df,
        train_path_map,
        TRAIN_IMAGES_DIR,
        transform=val_transform,
        is_train=False,
    )

    test_dataset = RSNADataset(
        test_df, test_path_map, TEST_IMAGES_DIR, transform=val_transform, is_train=False
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
