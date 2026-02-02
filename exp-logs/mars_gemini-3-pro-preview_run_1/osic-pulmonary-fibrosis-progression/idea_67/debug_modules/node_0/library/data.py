import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom

from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by instance number,
    and converts them to a 3D numpy array (Depth, Height, Width) in Hounsfield Units.
    """
    try:
        # Get all dicom files
        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            return None

        # Sort by instance number (filename) as proxy for Z-position
        # Filenames are like '1.dcm', '10.dcm', etc.
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        slices = []
        for f in files:
            ds = pydicom.dcmread(f)
            slices.append(ds)

        # Convert to Hounsfield Units
        images = []
        for s in slices:
            img_2d = s.pixel_array.astype(np.float32)

            # Apply Slope and Intercept if present
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", 0)

            if slope != 1:
                img_2d = slope * img_2d.astype(np.float64)
                img_2d = img_2d.astype(np.float32)

            img_2d += np.float32(intercept)
            images.append(img_2d)

        return np.stack(images)  # (D, H, W)

    except Exception as e:
        print(f"Error loading scan from {path}: {e}")
        return None


def generate_tri_slab(volume, axis=0, img_size=224):
    """
    Generates a 3-channel image (Tri-Slab) from a 3D volume using Maximum Intensity Projection (MIP).

    Args:
        volume (np.array): 3D array (D, H, W)
        axis (int): 0 for Axial (Top-down), 1 for Coronal (Front-back)
        img_size (int): Target resolution

    Returns:
        np.array: (H, W, 3) normalized image in range [0, 1]
    """
    # If Coronal, transpose to make Y the depth dimension: (H, D, W) -> (D', H', W')
    if axis == 1:
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # Handle small volumes
    if depth < 3:
        # Repeat volume to create fake depth
        volume = np.repeat(volume, 3, axis=0)
        depth = volume.shape[0]

    # Define slab boundaries with overlap
    # We want 3 slabs covering the volume with overlap
    # Slab 1: 0-33% + overlap
    # Slab 2: 33%-66% +/- overlap
    # Slab 3: 66%-100% - overlap

    p1 = depth // 3
    p2 = 2 * depth // 3
    overlap = int(depth * Config.SLAB_OVERLAP)

    # Indices
    idx_ranges = [
        (0, p1 + overlap),  # Slab 1
        (p1 - overlap, p2 + overlap),  # Slab 2
        (p2 - overlap, depth),  # Slab 3
    ]

    channels = []
    for start, end in idx_ranges:
        # Clip indices
        start = max(0, start)
        end = min(depth, end)
        if start >= end:
            start = max(0, end - 1)

        slab = volume[start:end, :, :]

        # MIP
        if slab.shape[0] > 0:
            mip = np.max(slab, axis=0)
        else:
            mip = np.zeros((volume.shape[1], volume.shape[2]), dtype=np.float32)

        channels.append(mip)

    # Stack to (H, W, 3)
    img = np.stack(channels, axis=-1)

    # Lung Windowing [-1000, 400] -> [-1250, 250] roughly covers lung
    # Standard Lung Window: Width 1500, Level -600 => -1350 to 150
    # Let's use a robust range for normalization
    min_hu = -1000.0
    max_hu = 400.0

    img = np.clip(img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)  # Normalize to 0-1

    # Resize
    img = cv2.resize(img, (img_size, img_size))

    return img.astype(np.float32)


# ==========================================
# Tabular Preprocessor
# ==========================================


class TabularPreprocessor:
    """
    Handles normalization and encoding of tabular features.
    Fits on training data, transforms all.
    """

    def __init__(self):
        self.age_stats = {"min": 0, "max": 100}
        self.pct_stats = {"min": 0, "max": 100}

    def fit(self, df):
        # Fit on Baseline features
        self.age_stats["min"] = df["Baseline_Age"].min()
        self.age_stats["max"] = df["Baseline_Age"].max()

        self.pct_stats["min"] = df["Baseline_Percent"].min()
        self.pct_stats["max"] = df["Baseline_Percent"].max()

    def transform(self, age, sex, smoking, percent):
        """
        Returns a vector: [Age_norm, Sex_binary, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_norm]
        """
        # Normalize Age
        age_norm = (age - self.age_stats["min"]) / (
            self.age_stats["max"] - self.age_stats["min"] + 1e-6
        )

        # Normalize Percent
        pct_norm = (percent - self.pct_stats["min"]) / (
            self.pct_stats["max"] - self.pct_stats["min"] + 1e-6
        )

        # Encode Sex (Male:0, Female:1)
        sex_val = 0.0 if sex == "Male" else 1.0

        # Encode Smoking (One-hot)
        # Categories: 'Ex-smoker', 'Never smoked', 'Currently smokes'
        smoke_ex = 1.0 if smoking == "Ex-smoker" else 0.0
        smoke_never = 1.0 if smoking == "Never smoked" else 0.0
        smoke_curr = 1.0 if smoking == "Currently smokes" else 0.0

        return np.array(
            [age_norm, sex_val, smoke_ex, smoke_never, smoke_curr, pct_norm],
            dtype=np.float32,
        )


# ==========================================
# Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, preprocessor=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.preprocessor = preprocessor

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial and Coronal)
        # ----------------------------------
        cache_path_ax = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
        cache_path_cor = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

        # Try loading from cache
        try:
            img_ax = np.load(cache_path_ax)
            img_cor = np.load(cache_path_cor)
        except (FileNotFoundError, ValueError, OSError):
            # Process from scratch
            dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
            volume = load_scan(dicom_dir)

            if volume is None:
                # Fallback for missing data: create black image
                img_ax = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                )
                img_cor = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                )
            else:
                img_ax = generate_tri_slab(volume, axis=0, img_size=Config.IMG_SIZE)
                img_cor = generate_tri_slab(volume, axis=1, img_size=Config.IMG_SIZE)

            # Save to cache
            np.save(cache_path_ax, img_ax)
            np.save(cache_path_cor, img_cor)

        # 2. Apply Augmentations
        # ----------------------
        # Albumentations expects HWC uint8 or float. Our images are float 0-1.
        if self.transform:
            # We apply the same spatial transform to both if possible,
            # but since they are different views, independent augmentation is acceptable/standard.
            res_ax = self.transform(image=img_ax)
            img_ax_t = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor_t = res_cor["image"]
        else:
            # Just convert to tensor
            t = ToTensorV2()
            img_ax_t = t(image=img_ax)["image"]
            img_cor_t = t(image=img_cor)["image"]

        # 3. Prepare Tabular Data
        # -----------------------
        # Use Baseline features
        tab_vec = self.preprocessor.transform(
            age=row["Baseline_Age"],
            sex=row["Baseline_Sex"],
            smoking=row["Baseline_SmokingStatus"],
            percent=row["Baseline_Percent"],
        )

        # 4. Prepare Target / Meta
        # ------------------------
        data = {
            "img_ax": img_ax_t,
            "img_cor": img_cor_t,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
        }

        if self.mode in ["train", "val"]:
            # Calculate Delta Week: Current Visit Week - Baseline Week
            delta_week = row["Weeks"] - row["Baseline_Weeks"]
            data["delta_week"] = torch.tensor(delta_week, dtype=torch.float32)
            data["target"] = torch.tensor(row["FVC"], dtype=torch.float32)

        elif self.mode == "test":
            # For test, we use Predict_Week - Baseline_Week
            delta_week = row["Predict_Week"] - row["Baseline_Week"]
            data["delta_week"] = torch.tensor(delta_week, dtype=torch.float32)
            data["patient_week"] = row["Patient_Week"]

        return data


# ==========================================
# Data Preparation & Loading
# ==========================================


def prepare_dataframe(df, is_train=True):
    """
    Prepares the dataframe by merging baseline features for training data.
    Test data structure is already handled by metadata generation.
    """
    if is_train:
        # For training data, we need to identify the baseline visit for each patient
        # and broadcast those features to all visits.

        # Sort to ensure we can pick the first visit (Baseline)
        df = df.sort_values(["Patient", "Weeks"])

        # Extract baseline rows (first visit per patient)
        baseline_df = df.groupby("Patient").first().reset_index()

        # Select columns to merge back
        cols = ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
        baseline_df = baseline_df[cols]

        # Rename to Baseline_
        rename_map = {c: f"Baseline_{c}" for c in cols if c != "Patient"}
        baseline_df = baseline_df.rename(columns=rename_map)

        # Merge back to original df
        merged = pd.merge(df, baseline_df, on="Patient", how="left")

    else:
        # Test/Val metadata already has specific structure or we treat Val like Train
        # If input is val.csv from metadata, it has same structure as train.csv
        # If input is test.csv from metadata, it already has Baseline_ columns

        if "Baseline_Weeks" not in df.columns and "Baseline_Week" not in df.columns:
            # This is likely the validation set from metadata (same schema as train)
            return prepare_dataframe(df, is_train=True)

        merged = df.copy()

    return merged


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Main function to create DataLoaders for Train, Val, and Test.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare Dataframes (Merge Baseline info)
    train_df = prepare_dataframe(train_df, is_train=True)
    val_df = prepare_dataframe(val_df, is_train=True)
    # Test df already has Baseline columns from metadata generation

    # Debug mode: subset data
    if Config.DEBUG:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        test_df = test_df.head(20)

    # 3. Initialize Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    # 4. Define Transforms
    # Spatial only, no intensity changes
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ]
    )

    # 5. Create Datasets
    train_ds = OSICDataset(
        train_df, mode="train", transform=train_transform, preprocessor=preprocessor
    )
    val_ds = OSICDataset(
        val_df, mode="val", transform=val_transform, preprocessor=preprocessor
    )
    test_ds = OSICDataset(
        test_df, mode="test", transform=val_transform, preprocessor=preprocessor
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
