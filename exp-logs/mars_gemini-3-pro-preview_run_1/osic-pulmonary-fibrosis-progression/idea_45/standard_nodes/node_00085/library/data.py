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

# ==========================================
# 1. Image Processing & Caching
# ==========================================


def load_dicom_volume(dicom_dir):
    """
    Loads DICOM files from a directory, sorts them by SliceLocation/InstanceNumber,
    and converts to Hounsfield Units.
    """
    full_path = Config.get_dicom_path(dicom_dir)
    if not os.path.exists(full_path):
        # Fallback for missing directories (should not happen based on EDA)
        return np.zeros((10, 512, 512), dtype=np.int16)

    files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, 512, 512), dtype=np.int16)

    # Read all files
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(full_path, f))
            slices.append(ds)
        except:
            continue

    if not slices:
        return np.zeros((10, 512, 512), dtype=np.int16)

    # Sort by ImagePositionPatient Z or InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Convert to HU
    images = []
    for s in slices:
        try:
            img = s.pixel_array.astype(np.float32)
        except RuntimeError:
            continue
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        img = img * slope + intercept
        images.append(img)

    if not images:
        return np.zeros((10, 512, 512), dtype=np.int16)

    volume = np.stack(images)  # (D, H, W)
    return volume


def generate_tri_slab(volume, axis=0, img_size=224, overlap=0.15):
    """
    Generates a 3-channel image using Fixed Overlapping Tri-Slabs.

    Args:
        volume: 3D numpy array (D, H, W)
        axis: 0 for Axial (split D), 1 for Coronal (split H)
        img_size: Target resolution
        overlap: Fraction of overlap
    """
    # Permute volume so the splitting axis is 0
    if axis == 1:
        # Coronal: (D, H, W) -> (H, D, W)
        # We want to split along H (Anterior-Posterior)
        vol = np.transpose(volume, (1, 0, 2))
    else:
        # Axial: (D, H, W)
        vol = volume

    depth = vol.shape[0]

    # Handle small volumes
    if depth < 3:
        # Repeat to fill
        indices = np.linspace(0, depth - 1, 3).astype(int)
        slabs = [vol[i : i + 1] for i in indices]
    else:
        # Define slab boundaries
        slab_h = depth / 3.0
        overlap_px = slab_h * overlap

        # Slab 1: 0 to 1/3 + overlap
        s1_end = int(slab_h + overlap_px)
        slab1 = vol[0:s1_end]

        # Slab 2: 1/3 - overlap to 2/3 + overlap
        s2_start = int(slab_h - overlap_px)
        s2_end = int(2 * slab_h + overlap_px)
        slab2 = vol[s2_start:s2_end]

        # Slab 3: 2/3 - overlap to end
        s3_start = int(2 * slab_h - overlap_px)
        slab3 = vol[s3_start:]

        slabs = [slab1, slab2, slab3]

    # Compute MIP and Resize
    channels = []
    for slab in slabs:
        if slab.shape[0] == 0:
            mip = np.zeros((img_size, img_size), dtype=np.float32)
        else:
            mip = np.max(slab, axis=0)

        # Normalize to 0-255 for resizing
        # HU Windowing: Lung Window [-1000, 400] roughly
        # We use min-max per image for robustness or fixed window?
        # Lesson 5 suggests disabling brightness augs, but normalization is needed.
        # Let's use a robust min-max normalization per slab to preserve relative structure
        v_min, v_max = -1000, 400
        mip = np.clip(mip, v_min, v_max)
        mip = (mip - v_min) / (v_max - v_min)
        mip = (mip * 255).astype(np.uint8)

        # Resize
        mip = cv2.resize(mip, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        channels.append(mip)

    # Stack to (H, W, 3)
    img = np.stack(channels, axis=-1)
    return img


def process_patient_images(patient_id, dicom_dir, load_cached_data=True):
    """
    Handles caching and processing of patient images.
    Returns:
        axial_img: (224, 224, 3) uint8
        coronal_img: (224, 224, 3) uint8
    """
    axial_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Fallback to re-process

    # 2. Process from Scratch
    volume = load_dicom_volume(dicom_dir)

    # Axial (Axis 0)
    axial = generate_tri_slab(
        volume, axis=0, img_size=Config.IMG_SIZE, overlap=Config.SLAB_OVERLAP
    )

    # Coronal (Axis 1)
    coronal = generate_tri_slab(
        volume, axis=1, img_size=Config.IMG_SIZE, overlap=Config.SLAB_OVERLAP
    )

    # 3. Save Cache
    np.save(axial_path, axial)
    np.save(coronal_path, coronal)

    return axial, coronal


# ==========================================
# 2. Tabular Preprocessing
# ==========================================


class TabularPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.stats = {}

    def fit(self, df):
        # Compute stats for normalization
        # Weeks: We use Baseline Weeks (0) for input, so this stat might be trivial,
        # but we compute it on the raw column just in case.
        # However, for Age and Percent, we need valid stats.

        # Numerical
        for col in ["Age", "Percent", "Weeks"]:
            mean = df[col].mean()
            std = df[col].std() + 1e-6
            self.stats[col] = {"mean": mean, "std": std}

    def transform(self, row):
        # Returns a list of features
        # Features: Weeks, Percent, Age, Sex, SmokingStatus (3)

        # 1. Numerical
        feats = []
        for col in ["Weeks", "Percent", "Age"]:
            val = row[col]
            stats = self.stats.get(col, {"mean": 0, "std": 1})
            norm_val = (val - stats["mean"]) / stats["std"]
            feats.append(norm_val)

        # 2. Sex (Binary)
        # Male: 0, Female: 1
        sex = 1.0 if row["Sex"] == "Female" else 0.0
        feats.append(sex)

        # 3. SmokingStatus (One-Hot: 3 dims)
        # Order: Ex-smoker, Never smoked, Currently smokes
        status = row["SmokingStatus"]
        if status == "Ex-smoker":
            feats.extend([1.0, 0.0, 0.0])
        elif status == "Never smoked":
            feats.extend([0.0, 1.0, 0.0])
        else:  # Currently smokes
            feats.extend([0.0, 0.0, 1.0])

        return np.array(feats, dtype=np.float32)


# ==========================================
# 3. Dataset Class
# ==========================================


class CTDataset(Dataset):
    def __init__(self, df, mode="train", preprocessor=None, load_cached=True):
        self.df = df.copy()
        self.mode = mode
        self.preprocessor = preprocessor
        self.load_cached = load_cached

        # Define Augmentations (Spatial Only)
        if mode == "train":
            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.Normalize(
                        mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.aug = A.Compose(
                [
                    A.Normalize(
                        mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ]
            )

        # Prepare Samples
        # We need to map every row to:
        # 1. Patient ID (for image)
        # 2. Baseline Tabular Features (for model input)
        # 3. Target FVC & dt (for loss)

        self.samples = []

        # Group by patient to find baseline
        if "Baseline_FVC" in self.df.columns:
            # Test set structure
            for idx, row in self.df.iterrows():
                # Input features come from Baseline_* columns
                # We construct a dict that mimics the train structure for the preprocessor
                input_feats = {
                    "Weeks": 0,  # Baseline relative week is 0
                    "Percent": row["Baseline_Percent"],
                    "Age": row["Baseline_Age"],
                    "Sex": row["Baseline_Sex"],
                    "SmokingStatus": row["Baseline_SmokingStatus"],
                }

                # dt = Predict_Week - Baseline_Week
                dt = row["Predict_Week"] - row["Baseline_Week"]

                self.samples.append(
                    {
                        "patient_id": row["Patient"],
                        "dicom_dir": row["dicom_dir"],
                        "input_feats": input_feats,
                        "target_fvc": 2000,  # Placeholder
                        "dt": dt,
                        "baseline_fvc": row["Baseline_FVC"],
                        "patient_week_id": row["Patient_Week"],
                    }
                )
        else:
            # Train/Val structure
            # We need to identify baseline for each patient
            patients = self.df["Patient"].unique()
            for pid in patients:
                p_data = self.df[self.df["Patient"] == pid]

                # Find baseline row (Weeks closest to 0)
                # Note: Some patients might not have exactly 0, take min abs
                p_data["abs_weeks"] = p_data["Weeks"].abs()
                baseline_row = p_data.loc[p_data["abs_weeks"].idxmin()]

                baseline_fvc = baseline_row["FVC"]
                baseline_week = baseline_row["Weeks"]

                # Construct Input Features (Static for this patient)
                input_feats = {
                    "Weeks": 0,  # Normalized Baseline Week
                    "Percent": baseline_row["Percent"],
                    "Age": baseline_row["Age"],
                    "Sex": baseline_row["Sex"],
                    "SmokingStatus": baseline_row["SmokingStatus"],
                }

                # Create samples for all visits
                for _, row in p_data.iterrows():
                    dt = row["Weeks"] - baseline_week
                    self.samples.append(
                        {
                            "patient_id": pid,
                            "dicom_dir": row["dicom_dir"],
                            "input_feats": input_feats,
                            "target_fvc": row["FVC"],
                            "dt": dt,
                            "baseline_fvc": baseline_fvc,
                            "patient_week_id": f"{pid}_{row['Weeks']}",
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load Images
        axial, coronal = process_patient_images(
            sample["patient_id"], sample["dicom_dir"], load_cached_data=self.load_cached
        )

        # 2. Augment
        # Apply same spatial augs to both? Or independent?
        # Independent is fine for dual-path, they are different views.
        aug_axial = self.aug(image=axial)["image"]
        aug_coronal = self.aug(image=coronal)["image"]

        # 3. Tabular Processing
        if self.preprocessor:
            tab_vec = self.preprocessor.transform(sample["input_feats"])
        else:
            tab_vec = np.zeros(7, dtype=np.float32)

        # 4. Prepare Tensors
        return {
            "axial": aug_axial,
            "coronal": aug_coronal,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "target": torch.tensor(sample["target_fvc"], dtype=torch.float32),
            "dt": torch.tensor(sample["dt"], dtype=torch.float32),
            "baseline_fvc": torch.tensor(sample["baseline_fvc"], dtype=torch.float32),
            "patient_week_id": sample["patient_week_id"],
        }


# ==========================================
# 4. Data Loaders
# ==========================================


def get_dataloaders(debug=False):
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        test_df = test_df.head(20)

    # Initialize Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    # Create Datasets
    train_ds = CTDataset(train_df, mode="train", preprocessor=preprocessor)
    val_ds = CTDataset(val_df, mode="val", preprocessor=preprocessor)
    test_ds = CTDataset(test_df, mode="test", preprocessor=preprocessor)

    # Create Loaders
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
