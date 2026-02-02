import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Try importing pydicom, handle if missing (though essential for the task)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    warnings.warn("pydicom not found. DICOM loading will fail or use fallback.")

from library.config import Config

# ==========================================
# Image Processing Functions
# ==========================================


def get_img_3d(path):
    """
    Loads DICOM files from a directory and constructs a 3D volume (Z, Y, X).
    Converts to Hounsfield Units and applies Lung Windowing.
    """
    if not HAS_PYDICOM:
        # Fallback for environments without pydicom (strictly for debugging flow)
        return np.zeros((20, 512, 512), dtype=np.float32)

    try:
        files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        if not files:
            return np.zeros((20, 512, 512), dtype=np.float32)

        slices = [pydicom.dcmread(os.path.join(path, f)) for f in files]
        # Sort by ImagePositionPatient Z coordinate
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Extract pixel data
        images = np.stack([s.pixel_array.astype(np.float32) for s in slices])

        # Convert to Hounsfield Units (HU)
        # Handle slope/intercept
        if hasattr(slices[0], "RescaleIntercept") and hasattr(
            slices[0], "RescaleSlope"
        ):
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope
            images = images * slope + intercept

        # Apply Lung Windowing [-1000, 400]
        # We clip to this range and normalize to 0-1
        min_hu = -1000.0
        max_hu = 400.0
        images = np.clip(images, min_hu, max_hu)
        images = (images - min_hu) / (max_hu - min_hu)

        return images  # Shape: (Z, Y, X), Range: [0, 1]

    except Exception as e:
        print(f"Error processing DICOMs in {path}: {e}")
        return np.zeros((20, 512, 512), dtype=np.float32)


def generate_tri_slab(volume, view="axial", img_size=224):
    """
    Generates Fixed Overlapping Tri-Slab MIPs.

    Args:
        volume: 3D numpy array (Z, Y, X)
        view: 'axial' or 'coronal'
        img_size: Output spatial dimension

    Returns:
        numpy array (img_size, img_size, 3) normalized to 0-1
    """
    # Determine dimensions based on view
    # Axial: Project along Z (dim 0). Image plane is (Y, X).
    # Coronal: Project along Y (dim 1). Image plane is (Z, X).

    if view == "axial":
        # Volume is (D, H, W)
        D = volume.shape[0]
        projection_axis = 0
    elif view == "coronal":
        # Volume is (Z, D, W). We want to project along Y (dim 1).
        D = volume.shape[1]
        projection_axis = 1
    else:
        raise ValueError("View must be 'axial' or 'coronal'")

    if D < 3:
        # Handle edge case with very few slices by repeating
        # Return a simple max projection repeated 3 times
        mip = np.max(volume, axis=projection_axis)
        mip = cv2.resize(mip, (img_size, img_size))
        return np.stack([mip] * 3, axis=-1)

    # Define Slab Boundaries with 15% overlap
    # We want 3 slabs covering [0, D]
    # Length of one slab roughly D/3.
    # Overlap amount
    overlap = int(D * 0.15)

    # Slab 1: 0 to 33% + overlap
    end1 = int(D / 3) + overlap

    # Slab 2: 33% - overlap to 66% + overlap
    start2 = int(D / 3) - overlap
    end2 = int(2 * D / 3) + overlap

    # Slab 3: 66% - overlap to 100%
    start3 = int(2 * D / 3) - overlap

    # Clip indices
    start2 = max(0, start2)
    start3 = max(0, start3)
    end1 = min(D, end1)
    end2 = min(D, end2)

    # Extract Slabs
    if view == "axial":
        slab1 = volume[0:end1, :, :]
        slab2 = volume[start2:end2, :, :]
        slab3 = volume[start3:, :, :]
    else:  # coronal
        slab1 = volume[:, 0:end1, :]
        slab2 = volume[:, start2:end2, :]
        slab3 = volume[:, start3:, :]

    # Compute MIPs (Maximum Intensity Projection)
    # Resulting shape for Axial: (Y, X)
    # Resulting shape for Coronal: (Z, X)
    m1 = np.max(slab1, axis=projection_axis)
    m2 = np.max(slab2, axis=projection_axis)
    m3 = np.max(slab3, axis=projection_axis)

    # Resize to target resolution
    # cv2.resize expects (W, H)
    m1 = cv2.resize(m1, (img_size, img_size))
    m2 = cv2.resize(m2, (img_size, img_size))
    m3 = cv2.resize(m3, (img_size, img_size))

    # Stack to create RGB-like channels
    img = np.stack([m1, m2, m3], axis=-1)

    return img


def process_patient_images(
    patient_id, dicom_rel_path, cache_dir, load_cached_data=True
):
    """
    Orchestrates image loading, processing, and caching.
    Returns (axial_img, coronal_img).
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            ax = np.load(axial_path)
            cor = np.load(coronal_path)
            return ax, cor
        except:
            pass  # Fallback to re-processing

    # 2. Process from scratch
    full_dicom_path = os.path.join(Config.INPUT_DIR, dicom_rel_path)
    volume = get_img_3d(full_dicom_path)

    ax = generate_tri_slab(volume, view="axial", img_size=Config.IMAGE_SIZE)
    cor = generate_tri_slab(volume, view="coronal", img_size=Config.IMAGE_SIZE)

    # 3. Save to Cache
    np.save(axial_path, ax)
    np.save(coronal_path, cor)

    return ax, cor


# ==========================================
# Tabular Preprocessing
# ==========================================


class TabularPreprocessor:
    def __init__(self):
        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.num_cols = ["Age", "Percent"]
        self.cat_cols = ["Sex", "SmokingStatus"]

    def fit(self, df):
        self.num_scaler.fit(df[self.num_cols])
        self.cat_encoder.fit(df[self.cat_cols])

    def transform(self, df):
        num_feats = self.num_scaler.transform(df[self.num_cols])
        cat_feats = self.cat_encoder.transform(df[self.cat_cols])
        return np.concatenate([num_feats, cat_feats], axis=1)


# ==========================================
# Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        tabular_preprocessor,
        mode="train",
        transform=None,
        cache_dir=Config.CACHE_DIR,
    ):
        self.df = df.reset_index(drop=True)
        self.preprocessor = tabular_preprocessor
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir

        # Pre-compute tabular features
        # For training, we need to map the baseline columns to the standard columns
        # For test, we map 'Baseline_Age' -> 'Age', etc.

        process_df = self.df.copy()

        if mode == "test":
            # Rename baseline columns to match training schema for the preprocessor
            process_df = process_df.rename(
                columns={
                    "Baseline_Age": "Age",
                    "Baseline_Percent": "Percent",
                    "Baseline_Sex": "Sex",
                    "Baseline_SmokingStatus": "SmokingStatus",
                }
            )
        else:
            # For training/val, we must use Baseline features to match inference logic
            # (i.e., don't use the Percent from the future visit)
            # We assume the dataframe passed here already has 'Age', 'Percent' etc. populated
            # with the BASELINE values for that patient.
            pass

        self.tabular_features = self.preprocessor.transform(process_df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        # Use dicom_dir from metadata
        dicom_dir = row["dicom_dir"]

        # Load cached or process
        img_axial, img_coronal = process_patient_images(
            patient_id, dicom_dir, self.cache_dir, load_cached_data=True
        )

        # 2. Augmentations
        if self.transform:
            # Apply same spatial transform to both?
            # Independent augmentation is safer for independent backbones
            # Albumentations expects uint8 or float32. Our imgs are float32 [0,1]

            res_ax = self.transform(image=img_axial)["image"]
            res_cor = self.transform(image=img_coronal)["image"]
            img_axial = res_ax
            img_coronal = res_cor
        else:
            # Just convert to tensor
            t = ToTensorV2()
            img_axial = t(image=img_axial)["image"]
            img_coronal = t(image=img_coronal)["image"]

        # 3. Tabular
        tab_vec = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # 4. Target & Metadata
        # Target: FVC
        # Metadata: Weeks, Baseline_FVC

        if self.mode == "test":
            # In test mode, FVC is dummy, we predict it
            fvc = 0.0
            weeks = row["Predict_Week"]
            base_fvc = row["Baseline_FVC"]
            base_week = row["Baseline_Week"]
        else:
            fvc = row["FVC"]
            weeks = row["Weeks"]
            # For training, we need Baseline_FVC.
            # If it's not in the row, we assume the caller prepared it.
            # In our prepare_data logic below, we will ensure it exists.
            base_fvc = row["Baseline_FVC"]
            base_week = row["Baseline_Week"]

        return {
            "img_axial": img_axial,  # (3, 224, 224)
            "img_coronal": img_coronal,  # (3, 224, 224)
            "tabular": tab_vec,  # (D_tab,)
            "target": torch.tensor(fvc, dtype=torch.float32),
            "weeks": torch.tensor(weeks, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "base_week": torch.tensor(base_week, dtype=torch.float32),
            "patient_id": patient_id,
        }


# ==========================================
# Data Preparation & Loader Factory
# ==========================================


def prepare_training_data(train_df, val_df):
    """
    Enhances training data by extracting baseline features for every patient
    and broadcasting them to all visits. This ensures training matches inference.
    """
    # Combine to find global baselines per patient
    all_df = pd.concat([train_df, val_df], axis=0)

    # Identify baseline row: min absolute weeks
    all_df["abs_weeks"] = all_df["Weeks"].abs()
    baseline_df = (
        all_df.sort_values(["Patient", "abs_weeks"])
        .groupby("Patient")
        .first()
        .reset_index()
    )

    # Select baseline columns
    baseline_cols = [
        "Patient",
        "FVC",
        "Weeks",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
    baseline_df = baseline_df[baseline_cols]

    # Rename to Baseline_X
    baseline_df.columns = [
        "Patient",
        "Baseline_FVC",
        "Baseline_Week",
        "Baseline_Percent",
        "Baseline_Age",
        "Baseline_Sex",
        "Baseline_SmokingStatus",
    ]

    # Merge back to original dfs
    # Note: We replace the varying Age/Percent/Sex/Smoking with the Baseline versions for the input features
    # But we keep the original FVC/Weeks as targets/metadata

    def merge_baseline(df):
        merged = pd.merge(df, baseline_df, on="Patient", how="left")
        # Overwrite feature columns with baseline values for consistency with test time
        merged["Age"] = merged["Baseline_Age"]
        merged["Percent"] = merged["Baseline_Percent"]
        merged["Sex"] = merged["Baseline_Sex"]
        merged["SmokingStatus"] = merged["Baseline_SmokingStatus"]
        return merged

    train_df_aug = merge_baseline(train_df)
    val_df_aug = merge_baseline(val_df)

    return train_df_aug, val_df_aug


def get_dataloaders(
    debug=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create dataloaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare Data (Baseline Extraction)
    train_df, val_df = prepare_training_data(train_df, val_df)

    # 3. Fit Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)  # Fit only on train

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        test_df = test_df.head(Config.DEBUG_SAMPLES)

    # 4. Define Augmentations (Spatial Only)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD), ToTensorV2()]
    )

    # 5. Create Datasets
    train_ds = OSICDataset(
        train_df, preprocessor, mode="train", transform=train_transform
    )
    val_ds = OSICDataset(val_df, preprocessor, mode="val", transform=val_transform)
    test_ds = OSICDataset(test_df, preprocessor, mode="test", transform=val_transform)

    # 6. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
