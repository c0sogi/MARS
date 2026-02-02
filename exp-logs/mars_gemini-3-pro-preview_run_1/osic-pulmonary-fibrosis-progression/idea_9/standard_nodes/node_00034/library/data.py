import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom
from library.config import Config

# -----------------------------------------------------------------------------
# Helper Functions for DICOM and Image Processing
# -----------------------------------------------------------------------------


def load_dicom_volume(dicom_dir, img_size=224):
    """
    Loads a DICOM series from a directory, converts to HU, and stacks into a 3D volume.
    Resizes slices to img_size x img_size to save memory.
    """
    if not os.path.exists(dicom_dir):
        return np.zeros((10, img_size, img_size), dtype=np.float32)

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, img_size, img_size), dtype=np.float32)

    # Read DICOMs
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(os.path.join(dicom_dir, f))
            slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return np.zeros((10, img_size, img_size), dtype=np.float32)

    # Sort by ImagePositionPatient Z-coordinate
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback: sort by InstanceNumber
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            slices.sort(key=lambda x: x.filename)

    # Convert to HU and resize
    processed_slices = []
    for s in slices:
        try:
            # Get pixel array
            img = s.pixel_array.astype(np.float32)

            # Convert to HU
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = img * slope + intercept

            # Resize
            img = cv2.resize(img, (img_size, img_size))
            processed_slices.append(img)
        except Exception:
            continue

    if not processed_slices:
        return np.zeros((10, img_size, img_size), dtype=np.float32)

    volume = np.stack(processed_slices, axis=0)  # (D, H, W)
    return volume


def generate_tri_slab_view(volume, axis=0, num_slabs=3, overlap=0.15):
    """
    Generates a 3-channel image using overlapping Maximum Intensity Projections (MIP).

    Args:
        volume: 3D numpy array (D, H, W)
        axis: 0 for Axial (along D), 1 for Coronal (along H)
        num_slabs: Number of slabs (channels)
        overlap: Fraction of overlap between slabs
    """
    # Determine the dimension size along the projection axis
    if axis == 0:
        depth = volume.shape[0]
    elif axis == 1:
        depth = volume.shape[1]
    else:
        raise ValueError("Axis must be 0 (Axial) or 1 (Coronal)")

    # Calculate slab parameters
    # Formula: L = S + (N-1)*(S - overlap*S)
    denom = 1 + (num_slabs - 1) * (1 - overlap)
    slab_size = max(1.0, depth / denom)
    stride = slab_size * (1 - overlap)

    channels = []
    for i in range(num_slabs):
        start = int(i * stride)
        end = int(start + slab_size)

        # Bounds check
        end = min(end, depth)
        if start >= end:
            start = max(0, end - 1)

        # Extract slab and compute MIP
        if axis == 0:
            # Axial: Slice along D (0), MIP along D
            slab = volume[start:end, :, :]
            if slab.shape[0] > 0:
                mip = np.max(slab, axis=0)
            else:
                mip = np.zeros((volume.shape[1], volume.shape[2]), dtype=volume.dtype)
        else:
            # Coronal: Slice along H (1), MIP along H
            slab = volume[:, start:end, :]
            if slab.shape[1] > 0:
                mip = np.max(slab, axis=1)  # Result is (D, W)
                # Resize to target square resolution (H, W) -> (224, 224)
                # Here we map the variable Depth to Height 224
                mip = cv2.resize(mip, (volume.shape[2], volume.shape[1]))
            else:
                mip = np.zeros((volume.shape[1], volume.shape[2]), dtype=volume.dtype)

        channels.append(mip)

    # Stack channels
    img = np.stack(channels, axis=-1)  # (H, W, 3)

    # Normalize HU to 0-255
    # Lung Window: [-1000, 400]
    min_hu = -1000
    max_hu = 400
    img = np.clip(img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)
    img = (img * 255).astype(np.uint8)

    return img


def get_patient_data(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Retrieves processed Axial and Coronal views. Implements caching.
    """
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    if load_cached and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception:
            pass  # Fallback to re-compute

    # Compute
    volume = load_dicom_volume(dicom_dir, img_size=Config.IMG_SIZE)

    # Axial View (Z-axis projection)
    img_ax = generate_tri_slab_view(
        volume, axis=0, num_slabs=Config.NUM_SLABS, overlap=Config.SLAB_OVERLAP
    )

    # Coronal View (Y-axis projection)
    img_cor = generate_tri_slab_view(
        volume, axis=1, num_slabs=Config.NUM_SLABS, overlap=Config.SLAB_OVERLAP
    )

    data = {"ax": img_ax, "cor": img_cor}

    # Save to cache
    try:
        np.save(cache_path, data)
    except Exception:
        pass

    return data


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class LungDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, cache_dir=Config.CACHE_DIR):
        self.df = df.copy()
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Mappings for categorical features
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Identify Baseline info for Training data
        if self.mode in ["train", "val"]:
            # Group by patient to find baseline (min absolute weeks or min weeks)
            self.patient_baselines = {}
            for pid, group in self.df.groupby("Patient"):
                # We assume the visit with min weeks is the baseline visit
                baseline_idx = group["Weeks"].idxmin()
                row = group.loc[baseline_idx]
                self.patient_baselines[pid] = {
                    "FVC": row["FVC"],
                    "Weeks": row["Weeks"],
                    "Percent": row["Percent"],
                    "Age": row["Age"],
                    "Sex": row["Sex"],
                    "SmokingStatus": row["SmokingStatus"],
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Get Images (Cached or Computed)
        # Construct full path to dicom directory
        dicom_path = os.path.join(Config.INPUT_DIR, row["dicom_dir"])

        images = get_patient_data(
            patient_id, dicom_path, self.cache_dir, load_cached=True
        )
        img_ax = images["ax"]
        img_cor = images["cor"]

        # 2. Augmentations
        if self.transform:
            # Apply augmentations independently to orthogonal views
            res_ax = self.transform(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor = res_cor["image"]
        else:
            # Fallback to ToTensor
            t = A.Compose([A.Normalize(), ToTensorV2()])
            img_ax = t(image=img_ax)["image"]
            img_cor = t(image=img_cor)["image"]

        # 3. Prepare Tabular & Target
        if self.mode in ["train", "val"]:
            # Retrieve baseline info
            base = self.patient_baselines[patient_id]

            base_fvc = base["FVC"]
            base_percent = base["Percent"]
            base_age = base["Age"]
            base_sex = self.sex_map.get(base["Sex"], 0)
            base_smoke = self.smoke_map.get(base["SmokingStatus"], 0)

            current_week = row["Weeks"]
            base_week = base["Weeks"]
            delta_week = current_week - base_week

            target_fvc = row["FVC"]

        else:  # Test mode
            # Columns: Baseline_FVC, Baseline_Percent, Baseline_Age, ...
            base_fvc = row["Baseline_FVC"]
            base_percent = row["Baseline_Percent"]
            base_age = row["Baseline_Age"]
            base_sex = self.sex_map.get(row["Baseline_Sex"], 0)
            base_smoke = self.smoke_map.get(row["Baseline_SmokingStatus"], 0)

            current_week = row["Predict_Week"]
            base_week = row["Baseline_Week"]
            delta_week = current_week - base_week

            target_fvc = 0.0  # Dummy

        # Normalize Tabular Features
        # Age: (x - 65) / 15
        norm_age = (base_age - 65.0) / 15.0
        # Percent: (x - 80) / 20
        norm_percent = (base_percent - 80.0) / 20.0
        # Base FVC Norm: (x - 2500) / 1000
        norm_base_fvc = (base_fvc - 2500.0) / 1000.0

        # One-hot encode Sex (2 classes) and Smoking (3 classes)
        sex_oh = [0.0, 0.0]
        if 0 <= base_sex < 2:
            sex_oh[base_sex] = 1.0

        smoke_oh = [0.0, 0.0, 0.0]
        if 0 <= base_smoke < 3:
            smoke_oh[base_smoke] = 1.0

        # Construct Tabular Vector: [Age, Percent, Base_FVC, Sex_0, Sex_1, Smoke_0, Smoke_1, Smoke_2]
        tabular_list = [norm_age, norm_percent, norm_base_fvc] + sex_oh + smoke_oh
        tabular = np.array(tabular_list, dtype=np.float32)

        return {
            "img_ax": img_ax.clone(),
            "img_cor": img_cor.clone(),
            "tabular": torch.from_numpy(tabular).clone(),
            "meta": torch.tensor([delta_week, base_fvc], dtype=torch.float32),
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{current_week}"
            ),
        }


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------


def get_dataloaders(debug=False):
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Define Transforms
    # ImageNet Normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Train: Spatial Augmentations Only (No Intensity changes)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # Val/Test: Resize (handled in dataset) + Normalize
    val_transform = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

    # Create Datasets
    train_ds = LungDataset(train_df, mode="train", transform=train_transform)
    val_ds = LungDataset(val_df, mode="val", transform=val_transform)
    test_ds = LungDataset(test_df, mode="test", transform=val_transform)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
