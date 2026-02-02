import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import configuration and utilities
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMG_SIZE,
    STRIDE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import seed_everything

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ==========================================
# Helper Functions
# ==========================================


def read_dicom_file(path):
    """
    Reads a DICOM file and returns the pixel array.
    Tries pydicom first, then OpenCV.
    """
    if not os.path.exists(path):
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Method 1: pydicom
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            return img
        except Exception:
            pass

    # Method 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # Fallback
    return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def load_modality_volume(base_path, modality_name):
    """
    Loads all DICOM files for a specific modality, sorts them by instance number,
    and returns a 3D numpy array (D, H, W).
    """
    modality_path = os.path.join(base_path, modality_name)
    if not os.path.exists(modality_path):
        return None

    files = [f for f in os.listdir(modality_path) if f.endswith(".dcm")]
    if not files:
        return None

    # Sort files. Usually filenames are Image-N.dcm.
    # We try to extract N for sorting.
    def extract_number(fname):
        s = "".join(filter(str.isdigit, fname))
        return int(s) if s.isdigit() else 0

    files.sort(key=extract_number)

    volume = []
    for f in files:
        f_path = os.path.join(modality_path, f)
        img = read_dicom_file(f_path)

        # Resize immediately to save memory if volume is large,
        # but we need original aspect for centroid?
        # No, resizing is fine as long as we do it consistently.
        # However, for centroid calculation, we just need to know if pixel > 0.
        # Let's resize to target size to keep memory usage predictable.
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        volume.append(img)

    return np.array(volume)  # (D, H, W)


def get_anatomical_centroid(volume):
    """
    Calculates the Z-axis geometric centroid of the brain tissue.
    Thresholds pixels > 0 to define ROI.
    Returns the index of the center slice.
    """
    if volume is None or len(volume) == 0:
        return 0

    # Threshold to find brain
    # Simple check: max pixel value might be high, so > 0 is usually sufficient for MRI background
    z_indices = np.where(np.any(volume > 0, axis=(1, 2)))[0]

    if len(z_indices) == 0:
        return len(volume) // 2

    # Geometric center of the bounding box along Z
    z_min = np.min(z_indices)
    z_max = np.max(z_indices)
    centroid = (z_min + z_max) // 2

    return int(centroid)


def process_subject(row, input_dir, stride=STRIDE):
    """
    Process a single subject:
    1. Load volumes for FLAIR, T1wCE, T2w.
    2. Find centroid for each.
    3. Extract slices at c-stride, c, c+stride.
    4. Stack into (H, W, 9).
    5. Normalize channels independently.
    """
    # row contains relative paths, e.g., "train/00000"
    # We need to construct full path
    subject_path = os.path.join(input_dir, row["subject_path"])

    # Modalities of interest for the 9-channel input
    # Order: FLAIR, T1wCE, T2w
    modalities = ["FLAIR", "T1wCE", "T2w"]

    # Cite solution_lesson_node_00009: Revert to 3-channel input (Middle Slice)
    # We collect 3 slices: [FLAIR, T1wCE, T2w] at z (centroid)
    channels = []

    # We process each modality
    for mod in modalities:
        vol = load_modality_volume(subject_path, mod)

        if vol is None:
            # Handle missing modality by filling with zeros
            channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
            continue

        centroid = get_anatomical_centroid(vol)
        # Extract middle slice only
        img = vol[centroid]
        channels.append(img)

    # Stack: (H, W, 3)
    img_stack = np.stack(channels, axis=-1)

    # Independent Channel Min-Max Scaling
    # Avoid division by zero
    for c in range(img_stack.shape[-1]):
        ch_min = img_stack[..., c].min()
        ch_max = img_stack[..., c].max()
        if ch_max > ch_min:
            img_stack[..., c] = (img_stack[..., c] - ch_min) / (ch_max - ch_min)
        else:
            img_stack[..., c] = 0.0

    return img_stack.astype(np.float32)


def generate_dataset_arrays(df, split_name, load_cached_data=True):
    """
    Generates or loads cached numpy arrays for the dataset.
    """
    cache_dir = WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"cache_{split_name}_ids.npy")
    images_path = os.path.join(cache_dir, f"cache_{split_name}_images.npy")
    targets_path = os.path.join(cache_dir, f"cache_{split_name}_targets.npy")

    # Check cache
    if load_cached_data:
        if os.path.exists(ids_path) and os.path.exists(images_path):
            # If targets exist or it's test set (targets might not exist)
            if split_name == "test" or os.path.exists(targets_path):
                print(f"Checking cached {split_name} data in {cache_dir}...")
                ids = np.load(ids_path)

                # Cite solution_lesson_node_00071: Verify Cache Integrity
                if len(ids) == len(df):
                    print(f"Cache hit: Loaded {len(ids)} samples.")
                    images = np.load(images_path)
                    if split_name != "test":
                        targets = np.load(targets_path)
                        return ids, images, targets
                    else:
                        return ids, images, None
                else:
                    print(
                        f"Cache mismatch: Found {len(ids)} samples, expected {len(df)}. Regenerating..."
                    )

    print(f"Processing {split_name} data (Cache miss or invalid)...")

    ids_list = []
    images_list = []
    targets_list = []

    # Debug mode
    if DEBUG:
        df = df.head(DEBUG_SAMPLE_SIZE).copy()

    total = len(df)
    for idx, row in df.iterrows():
        # Process image
        img_tensor = process_subject(row, INPUT_DIR)

        ids_list.append(row["BraTS21ID"])
        images_list.append(img_tensor)

        if "MGMT_value" in row:
            targets_list.append(row["MGMT_value"])

    # Convert to numpy
    ids_np = np.array(ids_list)
    images_np = np.array(images_list)  # (N, H, W, 9)

    # Save to cache
    np.save(ids_path, ids_np)
    np.save(images_path, images_np)

    if targets_list:
        targets_np = np.array(targets_list, dtype=np.float32)
        np.save(targets_path, targets_np)
        return ids_np, images_np, targets_np
    else:
        return ids_np, images_np, None


# ==========================================
# Dataset Class
# ==========================================


class AGIVDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, 9)
        image = self.images[idx]

        if self.transform:
            # Albumentations works on the image array.
            # Ensure it handles multi-channel correctly (it does by default for geometric transforms)
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor manually if no transform
            image = torch.tensor(image).permute(2, 0, 1)  # (C, H, W)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image, torch.tensor(0.0)  # Dummy label for test


# ==========================================
# Transforms
# ==========================================


def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose(
            [
                # Geometric augmentations only, preserving relative channel alignment
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                # NO Translation (Shift) or Scaling as per AGIV strategy
                # Convert to Tensor (C, H, W)
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# Main Data Loader Function
# ==========================================


def get_dataloaders(
    train_metadata_path, val_metadata_path, test_metadata_path, load_cached_data=True
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load Metadata CSVs
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)
    df_test = pd.read_csv(test_metadata_path)

    # Generate/Load Data Arrays
    # Train
    train_ids, train_images, train_targets = generate_dataset_arrays(
        df_train, "train", load_cached_data=load_cached_data
    )

    # Val
    val_ids, val_images, val_targets = generate_dataset_arrays(
        df_val, "val", load_cached_data=load_cached_data
    )

    # Test
    test_ids, test_images, _ = generate_dataset_arrays(
        df_test, "test", load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = AGIVDataset(
        train_images, train_targets, transform=get_transforms("train")
    )
    val_dataset = AGIVDataset(val_images, val_targets, transform=get_transforms("val"))
    test_dataset = AGIVDataset(test_images, None, transform=get_transforms("test"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
