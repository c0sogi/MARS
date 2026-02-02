import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Attempt to import pydicom
try:
    import pydicom
except ImportError:
    pydicom = None


def get_img_tri_slab(patient_id, dicom_dir, view="axial", load_cached_data=True):
    """
    Generates or loads a fixed overlapping orthogonal tri-slab (MIP) for a patient.

    Args:
        patient_id (str): Patient ID.
        dicom_dir (str): Path to directory containing DICOM files.
        view (str): 'axial' or 'coronal'.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        np.ndarray: (224, 224, 3) image in [0, 1] range.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_{view}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            img = np.load(cache_path)
            return img
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    # Return blank image if pydicom is missing or directory invalid
    if pydicom is None or not os.path.exists(dicom_dir):
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    files = [
        os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")
    ]
    if not files:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Load slices
    scans = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            scans.append(ds)
        except:
            continue

    if not scans:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Sort by Instance Number or Image Position
    try:
        scans.sort(key=lambda x: float(x.InstanceNumber))
    except AttributeError:
        try:
            scans.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            pass  # Keep original order if sorting fails

    # Convert to Hounsfield Units (HU)
    image_stack = []
    for s in scans:
        try:
            img_2d = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img_2d = img_2d * slope + intercept
            image_stack.append(img_2d)
        except:
            continue

    if not image_stack:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    volume = np.stack(image_stack)  # Shape: (D, H, W)

    # Lung Windowing: Level -600, Width 1500 -> Range [-1350, 150]
    volume = np.clip(volume, -1350, 150)

    # Normalize to [0, 1]
    volume = (volume - (-1350)) / (150 - (-1350))

    # Handle Views
    if view == "coronal":
        # Axial is (Depth, Height, Width).
        # Coronal projection is typically along the Anterior-Posterior axis.
        # Assuming D=Z, H=Y, W=X. Coronal view looks at Z-X plane, projecting through Y.
        # We transpose to (H, D, W) so the first dimension is the one we project through.
        volume = volume.transpose(1, 0, 2)

    # Generate Tri-Slab with Overlap
    # Volume shape is (N, H, W) where N is the depth dimension to project
    n_slices = volume.shape[0]

    # Define 3 slabs with ~15% overlap relative to total depth
    # Segments: 0-40%, 30-70%, 60-100%
    p1_end = int(n_slices * 0.40)
    p2_start = int(n_slices * 0.30)
    p2_end = int(n_slices * 0.70)
    p3_start = int(n_slices * 0.60)

    # Ensure indices are valid
    p1_end = max(1, p1_end)
    p2_start = min(p2_start, n_slices - 1)
    p2_end = max(p2_start + 1, p2_end)
    p3_start = min(p3_start, n_slices - 1)

    slab1 = volume[:p1_end, :, :]
    slab2 = volume[p2_start:p2_end, :, :]
    slab3 = volume[p3_start:, :, :]

    # Compute MIP (Maximum Intensity Projection)
    h, w = volume.shape[1], volume.shape[2]

    def get_mip(slab, shape):
        if slab.shape[0] == 0:
            return np.zeros(shape, dtype=np.float32)
        return np.max(slab, axis=0)

    mip1 = get_mip(slab1, (h, w))
    mip2 = get_mip(slab2, (h, w))
    mip3 = get_mip(slab3, (h, w))

    # Stack to RGB (3 channels)
    img_out = np.stack([mip1, mip2, mip3], axis=-1)  # (H, W, 3)

    # Resize to Config.IMG_SIZE (224x224)
    try:
        img_out = cv2.resize(img_out, (Config.IMG_SIZE, Config.IMG_SIZE))
    except:
        img_out = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Save to cache
    try:
        np.save(cache_path, img_out)
    except:
        pass

    return img_out


class LungDataset(Dataset):
    def __init__(self, df, tabular_stats=None, phase="train"):
        self.df = df.reset_index(drop=True)
        self.tabular_stats = tabular_stats
        self.phase = phase

        # Augmentations
        if phase == "train":
            self.transform = A.Compose(
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
            self.transform = A.Compose(
                [A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Images
        dicom_path = os.path.join(Config.INPUT_DIR, row["dicom_dir"])

        # Load Axial and Coronal (use cache)
        img_axial = get_img_tri_slab(
            patient_id, dicom_path, "axial", load_cached_data=True
        )
        img_coronal = get_img_tri_slab(
            patient_id, dicom_path, "coronal", load_cached_data=True
        )

        # Apply Transforms
        res_ax = self.transform(image=img_axial)["image"]
        res_cor = self.transform(image=img_coronal)["image"]

        # 2. Tabular Features
        # Features: Age, Percent (Normalized), Sex, Smoking (One-Hot)
        age = row["Age"] if "Age" in row else row["Baseline_Age"]
        percent = row["Percent"] if "Percent" in row else row["Baseline_Percent"]

        if self.tabular_stats:
            age = (age - self.tabular_stats["age_mean"]) / (
                self.tabular_stats["age_std"] + 1e-6
            )
            percent = (percent - self.tabular_stats["pct_mean"]) / (
                self.tabular_stats["pct_std"] + 1e-6
            )

        # Sex: Male=[1,0], Female=[0,1]
        sex = row["Sex"] if "Sex" in row else row["Baseline_Sex"]
        sex_vec = [1, 0] if sex == "Male" else [0, 1]

        # Smoking: Ex-smoker=[1,0,0], Never smoked=[0,1,0], Currently smokes=[0,0,1]
        smoke = (
            row["SmokingStatus"]
            if "SmokingStatus" in row
            else row["Baseline_SmokingStatus"]
        )
        if smoke == "Ex-smoker":
            smoke_vec = [1, 0, 0]
        elif smoke == "Never smoked":
            smoke_vec = [0, 1, 0]
        else:
            smoke_vec = [0, 0, 1]

        tab_vec = np.array([age, percent] + sex_vec + smoke_vec, dtype=np.float32)

        # 3. Targets and Meta
        base_fvc = row["Baseline_FVC"]

        # Determine current week and delta
        if "Weeks" in row:
            current_week = row["Weeks"]
        elif "Predict_Week" in row:
            current_week = row["Predict_Week"]
        else:
            current_week = 0

        base_week = row["Baseline_Week"] if "Baseline_Week" in row else 0
        delta_week = current_week - base_week

        inputs = {
            "axial": res_ax,
            "coronal": res_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "delta_week": torch.tensor(delta_week, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
        }

        target = 0.0
        if self.phase != "test" and "FVC" in row:
            target = row["FVC"]

        return inputs, torch.tensor(target, dtype=torch.float32)


def get_dataloaders():
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Helper to add baseline FVC info to train/val dataframes
    def add_baseline_info(df):
        patients = df["Patient"].unique()
        baseline_data = []
        for p in patients:
            p_data = df[df["Patient"] == p]
            # Find row closest to week 0 (baseline)
            # We create a temporary column for absolute week difference from 0
            p_data = p_data.copy()
            p_data["abs_week"] = p_data["Weeks"].abs()
            p_data = p_data.sort_values("abs_week")

            baseline_row = p_data.iloc[0]

            baseline_data.append(
                {
                    "Patient": p,
                    "Baseline_FVC": baseline_row["FVC"],
                    "Baseline_Week": baseline_row["Weeks"],
                }
            )

        base_df = pd.DataFrame(baseline_data)
        # Merge baseline info back to original dataframe
        df = pd.merge(df, base_df, on="Patient", how="left")
        return df

    # Enrich Train and Val data
    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # Compute Tabular Stats from Training Data
    tab_stats = {
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
        "pct_mean": train_df["Percent"].mean(),
        "pct_std": train_df["Percent"].std(),
    }

    # Create Datasets
    train_ds = LungDataset(train_df, tab_stats, phase="train")
    val_ds = LungDataset(val_df, tab_stats, phase="val")
    test_ds = LungDataset(test_df, tab_stats, phase="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
