import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Constants for Lung Windowing
WINDOW_MIN = -1000
WINDOW_MAX = 400


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by slice location.
    """
    slices = [pydicom.dcmread(p) for p in glob.glob(os.path.join(path, "*.dcm"))]
    if not slices:
        return []

    # Sort by ImagePositionPatient Z coordinate if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts DICOM slices to a Hounsfield Unit (HU) numpy array.
    Handles RescaleSlope and RescaleIntercept.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    for n, s in enumerate(slices):
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else -1024
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1

        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)

        image[n] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def process_volume(dicom_dir, target_size=224):
    """
    Loads DICOM, converts to HU, applies Lung Window, and resizes slices.
    Returns: 3D numpy array (Depth, H, W) normalized to uint8 [0, 255].
    """
    try:
        slices = load_scan(dicom_dir)
        if not slices:
            # Return empty volume if load fails
            return np.zeros((10, target_size, target_size), dtype=np.uint8)

        vol_hu = get_pixels_hu(slices)

        # Apply Lung Window [-1000, 400]
        vol_hu = np.clip(vol_hu, WINDOW_MIN, WINDOW_MAX)

        # Normalize to 0-1 then scale to 0-255
        vol_hu = (vol_hu - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
        vol_hu = (vol_hu * 255).astype(np.uint8)

        # Resize each slice to target_size
        resized_vol = []
        for i in range(vol_hu.shape[0]):
            img = vol_hu[i]
            if img.shape[0] != target_size or img.shape[1] != target_size:
                img = cv2.resize(
                    img, (target_size, target_size), interpolation=cv2.INTER_AREA
                )
            resized_vol.append(img)

        return np.stack(resized_vol)

    except Exception as e:
        print(f"Error processing {dicom_dir}: {e}")
        return np.zeros((10, target_size, target_size), dtype=np.uint8)


def preprocess_and_cache(df, cache_dir, input_root, load_cached_data=True):
    """
    Iterates through patients in the dataframe and caches their processed 3D volumes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    patients = df["Patient"].unique()

    print(f"Processing/Caching volumes for {len(patients)} patients in {cache_dir}...")

    for p in patients:
        save_path = os.path.join(cache_dir, f"{p}.npy")

        # If cache exists and we are allowed to use it, skip
        if load_cached_data and os.path.exists(save_path):
            continue

        # Process from scratch
        # 'dicom_dir' in metadata is relative, e.g., "train/ID..."
        rel_path = df[df["Patient"] == p]["dicom_dir"].iloc[0]
        full_path = os.path.join(input_root, rel_path)

        vol = process_volume(full_path, target_size=Config.IMG_SIZE)
        np.save(save_path, vol)


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for 2D images.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()])


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", transform=None):
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Prepare Tabular Data (Unify Train/Test formats)
        self._prepare_tabular_data()

    def _prepare_tabular_data(self):
        """
        Ensures the dataframe has Baseline_FVC, Baseline_Percent, etc.
        For training data, these are derived from the earliest visit.
        """
        if "Baseline_FVC" in self.df.columns:
            return  # Test set already has these columns

        # For Train/Val, derive baseline from the earliest visit (min Weeks)
        # We assume the row with the minimum 'Weeks' value is the baseline visit.
        # Note: Weeks can be negative, so we sort by numerical value.

        # Create a temporary DF with just baseline info per patient
        # Sort by Patient and Weeks to pick the first one
        sorted_df = self.df.sort_values(["Patient", "Weeks"])
        baseline_df = sorted_df.groupby("Patient").first().reset_index()

        # Select relevant columns
        cols_to_merge = ["Patient", "FVC", "Percent", "Age", "Weeks"]
        baseline_subset = baseline_df[cols_to_merge].rename(
            columns={
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Weeks": "Baseline_Week",
            }
        )

        # Merge back to main dataframe
        self.df = self.df.merge(baseline_subset, on="Patient", how="left")

    def __len__(self):
        return len(self.df)

    def _generate_stochastic_slabs(self, vol, axis=0):
        """
        Generates 3 MIP slabs with random boundaries (Depth Jitter).
        axis=0: Axial (Z-axis split)
        axis=1: Coronal (Y-axis split)
        """
        # If Coronal, permute Y to be the depth dimension (0)
        # Vol is (D, H, W)
        if axis == 1:
            vol = np.transpose(vol, (1, 0, 2))

        depth = vol.shape[0]
        # Pad if depth is too small
        if depth < 3:
            pad = np.zeros((3 - depth, vol.shape[1], vol.shape[2]), dtype=vol.dtype)
            vol = np.concatenate([vol, pad], axis=0)
            depth = 3

        # Sample boundaries
        b1_ratio = np.random.uniform(*Config.SLAB_BOUND1_RANGE)
        b2_ratio = np.random.uniform(*Config.SLAB_BOUND2_RANGE)

        idx1 = int(depth * b1_ratio)
        idx2 = int(depth * b2_ratio)

        # Enforce constraints
        idx1 = max(1, min(idx1, depth - 2))
        idx2 = max(idx1 + 1, min(idx2, depth - 1))

        # Slice and MIP
        m1 = np.max(vol[:idx1], axis=0)
        m2 = np.max(vol[idx1:idx2], axis=0)
        m3 = np.max(vol[idx2:], axis=0)

        return np.stack([m1, m2, m3])  # Shape (3, H, W)

    def _generate_fixed_slabs(self, vol, axis=0):
        """
        Generates 3 MIP slabs with fixed boundaries and overlap.
        """
        if axis == 1:
            vol = np.transpose(vol, (1, 0, 2))

        depth = vol.shape[0]
        if depth < 3:
            pad = np.zeros((3 - depth, vol.shape[1], vol.shape[2]), dtype=vol.dtype)
            vol = np.concatenate([vol, pad], axis=0)
            depth = 3

        # Divide into 3 roughly equal parts
        third = depth // 3
        margin = int(depth * Config.INFERENCE_OVERLAP)

        # Define centers
        c1, c2 = third, 2 * third

        # Slice with overlap
        # Slab 1: Start to c1 + margin
        s1 = vol[0 : min(depth, c1 + margin)]
        # Slab 2: c1 - margin to c2 + margin
        s2 = vol[max(0, c1 - margin) : min(depth, c2 + margin)]
        # Slab 3: c2 - margin to End
        s3 = vol[max(0, c2 - margin) :]

        # MIP
        m1 = np.max(s1, axis=0) if s1.size > 0 else np.zeros_like(vol[0])
        m2 = np.max(s2, axis=0) if s2.size > 0 else np.zeros_like(vol[0])
        m3 = np.max(s3, axis=0) if s3.size > 0 else np.zeros_like(vol[0])

        return np.stack([m1, m2, m3])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Cached Volume
        vol_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
        if os.path.exists(vol_path):
            vol = np.load(vol_path)
        else:
            # Fallback: create black volume
            vol = np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

        # 2. Generate Slabs (Axial & Coronal)
        if self.mode == "train":
            img_ax = self._generate_stochastic_slabs(vol, axis=0)
            img_cor = self._generate_stochastic_slabs(vol, axis=1)
        else:
            img_ax = self._generate_fixed_slabs(vol, axis=0)
            img_cor = self._generate_fixed_slabs(vol, axis=1)

        # 3. Apply Transforms
        # Transpose to (H, W, 3) for Albumentations
        img_ax = np.transpose(img_ax, (1, 2, 0))
        img_cor = np.transpose(img_cor, (1, 2, 0))

        # Resize to fixed dimensions (Cite debug_lesson_3)
        if img_ax.shape[0] != Config.IMG_SIZE or img_ax.shape[1] != Config.IMG_SIZE:
            img_ax = cv2.resize(
                img_ax, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )

        if img_cor.shape[0] != Config.IMG_SIZE or img_cor.shape[1] != Config.IMG_SIZE:
            img_cor = cv2.resize(
                img_cor,
                (Config.IMG_SIZE, Config.IMG_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        if self.transform:
            res_ax = self.transform(image=img_ax)["image"]
            res_cor = self.transform(image=img_cor)["image"]
        else:
            # Manual fallback
            res_ax = torch.tensor(img_ax).permute(2, 0, 1).float() / 255.0
            res_cor = torch.tensor(img_cor).permute(2, 0, 1).float() / 255.0

        # 4. Prepare Tabular Features
        # Extract features
        base_week = row["Baseline_Week"]
        base_pct = row["Baseline_Percent"]
        base_age = row["Baseline_Age"]
        base_fvc = row["Baseline_FVC"]

        # Handle column naming differences
        sex = row.get("Baseline_Sex", row.get("Sex"))
        smoke = row.get("Baseline_SmokingStatus", row.get("SmokingStatus"))

        # Determine current week
        if self.mode == "test":
            current_week = row["Predict_Week"]
        else:
            current_week = row["Weeks"]

        # Feature Engineering / Normalization
        delta_week = current_week - base_week

        # Simple scaling based on domain knowledge
        feat_week = delta_week / 100.0
        feat_pct = base_pct / 100.0
        feat_age = base_age / 100.0
        feat_sex = float(self.sex_map.get(sex, 0))
        feat_smoke = float(self.smoke_map.get(smoke, 0)) / 2.0  # 0, 0.5, 1.0

        tab_vec = torch.tensor(
            [feat_week, feat_pct, feat_age, feat_sex, feat_smoke], dtype=torch.float32
        )
        base_fvc_tensor = torch.tensor([base_fvc], dtype=torch.float32)

        # 5. Target
        if self.mode != "test":
            target = torch.tensor([row["FVC"]], dtype=torch.float32)
        else:
            target = torch.tensor([0.0], dtype=torch.float32)

        return {
            "img_ax": res_ax,
            "img_cor": res_cor,
            "tab": tab_vec,
            "base_fvc": base_fvc_tensor,
        }, target


def get_dataloaders(train_df, val_df, test_df):
    """
    Main entry point. Caches data and returns dataloaders.
    """
    # 1. Preprocess and Cache Data
    # Combine all patients to process in one go
    all_patients_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # Drop duplicates by Patient to avoid redundant processing
    unique_patients = all_patients_df.drop_duplicates(subset=["Patient"])

    preprocess_and_cache(
        unique_patients, Config.CACHE_DIR, Config.INPUT_ROOT, load_cached_data=True
    )

    # 2. Create Datasets
    train_ds = OSICDataset(
        train_df, Config.CACHE_DIR, mode="train", transform=get_transforms("train")
    )

    val_ds = OSICDataset(
        val_df, Config.CACHE_DIR, mode="val", transform=get_transforms("val")
    )

    test_ds = OSICDataset(
        test_df, Config.CACHE_DIR, mode="test", transform=get_transforms("test")
    )

    # 3. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
