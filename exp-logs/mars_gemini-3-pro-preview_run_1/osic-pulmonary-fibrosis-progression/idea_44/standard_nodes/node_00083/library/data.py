import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# Attempt to import pydicom; handle gracefully if missing (though required for .dcm)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def get_img_seq(dicom_dir):
    """
    Lists and sorts DICOM files from a directory.
    Sorts based on the integer value of the filename (e.g., '10.dcm' -> 10)
    to ensure correct anatomical ordering if InstanceNumber is missing.
    """
    files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return files


def load_scan(dicom_dir):
    """
    Loads a CT scan volume from a directory of DICOM files.
    Returns a numpy array of shape (D, H, W) in Hounsfield Units (HU).
    """
    if not HAS_PYDICOM:
        # If pydicom is strictly unavailable, we cannot process raw .dcm files.
        # We return a dummy volume to prevent crash, but this indicates an env issue.
        print(f"Warning: pydicom not found. Cannot load {dicom_dir}.")
        return np.zeros((10, 224, 224), dtype=np.float32)

    files = get_img_seq(dicom_dir)
    if not files:
        return np.zeros((10, 224, 224), dtype=np.float32)

    # Read DICOM files
    slices = [pydicom.dcmread(f) for f in files]

    # Sort slices by ImagePositionPatient[2] (Z-axis) if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass  # Fallback to filename sorting (already done)

    # Stack pixel arrays
    # Handle potential shape mismatches by resizing to the shape of the first slice
    try:
        image = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    except Exception:
        target_shape = slices[0].pixel_array.shape
        image_list = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            if img.shape != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]))
            image_list.append(img)
        image = np.stack(image_list)

    # Convert to Hounsfield Units (HU)
    # HU = pixel * slope + intercept
    if len(slices) > 0:
        slope = getattr(slices[0], "RescaleSlope", 1)
        intercept = getattr(slices[0], "RescaleIntercept", -1024)
        image = image * slope + intercept

    return image


def window_image(image, window_center=-600, window_width=1500):
    """
    Applies Lung Windowing to the CT scan and normalizes to [0, 1].
    Lung Window: W=1500, L=-600.
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = np.clip(image, img_min, img_max)

    # Normalize to [0, 1]
    if img_max - img_min != 0:
        window_image = (window_image - img_min) / (img_max - img_min)
    else:
        window_image = window_image - img_min

    return window_image


def generate_tri_slab(volume, view="axial", img_size=224):
    """
    Generates Fixed Overlapping Tri-Slabs (RGB) from a 3D volume.
    Splits the volume into 3 slabs with 15% overlap along the depth axis
    of the specified view, computes MIPs, and stacks them.

    Args:
        volume: 3D numpy array (D, H, W)
        view: 'axial' (D is depth) or 'coronal' (H is depth)
        img_size: Output spatial resolution

    Returns:
        numpy array (img_size, img_size, 3) normalized to [0, 1]
    """
    # Orient volume so dim 0 is the depth to split
    if view == "coronal":
        # Original: (D, H, W). Coronal view looks from Front (Y-axis/H).
        # Transpose to (H, D, W) so H is the depth
        vol_view = np.transpose(volume, (1, 0, 2))
    else:
        # Axial: (D, H, W). D is already depth.
        vol_view = volume

    depth = vol_view.shape[0]

    # Define Slab Boundaries (0-33%, 33-66%, 66-100%) with 15% overlap
    # Overlap is calculated as 15% of the total depth
    overlap = int(depth * 0.15)
    p33 = int(depth * 0.33)
    p66 = int(depth * 0.66)

    # Define indices with clamping
    s1_start, s1_end = 0, min(depth, p33 + overlap)
    s2_start, s2_end = max(0, p33 - overlap), min(depth, p66 + overlap)
    s3_start, s3_end = max(0, p66 - overlap), depth

    # Extract Slabs
    slab1 = vol_view[s1_start:s1_end, :, :]
    slab2 = vol_view[s2_start:s2_end, :, :]
    slab3 = vol_view[s3_start:s3_end, :, :]

    # Compute MIP (Maximum Intensity Projection)
    def get_mip(slab):
        if slab.shape[0] == 0:
            return np.zeros((vol_view.shape[1], vol_view.shape[2]), dtype=np.float32)
        return np.max(slab, axis=0)

    mip1 = get_mip(slab1)
    mip2 = get_mip(slab2)
    mip3 = get_mip(slab3)

    # Stack to RGB (H, W, 3)
    img_stack = np.stack([mip1, mip2, mip3], axis=-1)

    # Resize to target resolution
    img_resized = cv2.resize(img_stack, (img_size, img_size))

    return img_resized


def process_patient_scan(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Loads DICOM, processes into Axial and Coronal Tri-Slabs, and caches results.
    """
    cfg = Config()
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Loading Cached Data
    if load_cached and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Cache corrupted or load failed, recompute

    # 2. Compute from Scratch
    try:
        # Load and Window Volume
        volume = load_scan(dicom_dir)
        volume = window_image(volume)

        # Generate Views
        axial = generate_tri_slab(volume, view="axial", img_size=cfg.IMG_SIZE)
        coronal = generate_tri_slab(volume, view="coronal", img_size=cfg.IMG_SIZE)

        # 3. Save to Cache
        np.save(axial_path, axial)
        np.save(coronal_path, coronal)

        return axial, coronal

    except Exception as e:
        # Fallback for errors (e.g., corrupt DICOMs)
        print(f"Error processing {patient_id}: {e}")
        dummy = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)
        return dummy, dummy


class LungDataset(Dataset):
    def __init__(self, df, split="train", transform=None, debug=False):
        self.df = df.copy()
        self.split = split
        self.transform = transform
        self.cfg = Config()

        # Debug Mode: Reduce dataset size
        if debug:
            self.df = self.df.iloc[:50]

        # Pre-compute Baseline Features for Train/Val splits
        # In Train/Val, we have multiple rows per patient. We need the static baseline info.
        if split in ["train", "val"]:
            self.baseline_lookup = {}
            patients = self.df["Patient"].unique()
            for p in patients:
                # Get all records for this patient
                p_data = self.df[self.df["Patient"] == p]
                # Identify baseline row (min Weeks, usually 0 or negative)
                p_data = p_data.sort_values("Weeks")
                baseline_row = p_data.iloc[0]

                self.baseline_lookup[p] = {
                    "Age": baseline_row["Age"],
                    "Sex": baseline_row["Sex"],
                    "SmokingStatus": baseline_row["SmokingStatus"],
                    "Percent": baseline_row["Percent"],
                }

        # Mappings for Categorical Features
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image Data
        # Construct full path from relative path in metadata
        dicom_full_path = os.path.join(self.cfg.INPUT_ROOT, row["dicom_dir"])

        img_axial, img_coronal = process_patient_scan(
            patient_id, dicom_full_path, self.cfg.CACHE_DIR, self.cfg.LOAD_CACHED_DATA
        )

        # 2. Apply Augmentations (Spatial Only)
        if self.transform:
            # Apply transform independently to both views
            res_ax = self.transform(image=img_axial)
            img_axial = res_ax["image"]

            res_cor = self.transform(image=img_coronal)
            img_coronal = res_cor["image"]
        else:
            # Convert to Tensor (C, H, W)
            img_axial = torch.tensor(img_axial.transpose(2, 0, 1), dtype=torch.float32)
            img_coronal = torch.tensor(
                img_coronal.transpose(2, 0, 1), dtype=torch.float32
            )

        # 3. Extract Tabular Features
        if self.split in ["train", "val"]:
            base_feats = self.baseline_lookup[patient_id]
            age = base_feats["Age"]
            sex = base_feats["Sex"]
            smoke = base_feats["SmokingStatus"]
            percent = base_feats["Percent"]

            target_fvc = row["FVC"]
            current_week = row["Weeks"]
        else:
            # Test set metadata already contains baseline info columns
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
            percent = row["Baseline_Percent"]

            target_fvc = 0  # Placeholder for test
            current_week = row["Predict_Week"]

        # 4. Encode Tabular Features
        # Normalize continuous variables
        age_norm = age / 100.0
        perc_norm = percent / 100.0

        # One-Hot Encode Sex (2 dims)
        sex_vec = [0.0, 0.0]
        if sex in self.sex_map:
            sex_vec[self.sex_map[sex]] = 1.0

        # One-Hot Encode Smoking (3 dims)
        smoke_vec = [0.0, 0.0, 0.0]
        if smoke in self.smoke_map:
            smoke_vec[self.smoke_map[smoke]] = 1.0

        # Construct Feature Vector
        # [Age, Percent, Sex0, Sex1, Smoke0, Smoke1, Smoke2, Pad, Pad] -> Total 9 dims
        tab_vec = [age_norm, perc_norm] + sex_vec + smoke_vec + [0.0, 0.0]
        tab_tensor = torch.tensor(tab_vec, dtype=torch.float32)

        return {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "tabular": tab_tensor,
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "week": torch.tensor(current_week, dtype=torch.float32),
            "patient_id": patient_id,
        }


def get_dataloaders(cfg):
    """
    Creates and returns DataLoaders for Train, Validation, and Test sets.
    """
    # Load Metadata CSVs
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
    test_df = pd.read_csv(cfg.TEST_CSV)

    # Define Augmentations
    # Train: Spatial augmentations + Normalization
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Val/Test: Normalization only
    valid_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Create Datasets
    train_ds = LungDataset(
        train_df, split="train", transform=train_transform, debug=cfg.DEBUG
    )
    val_ds = LungDataset(
        val_df, split="val", transform=valid_transform, debug=cfg.DEBUG
    )
    test_ds = LungDataset(
        test_df, split="test", transform=valid_transform, debug=cfg.DEBUG
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
