import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything

# -------------------------------------------------------------------------
# DICOM Processing Helper Functions
# -------------------------------------------------------------------------


def load_scan(path):
    """Loads all DICOM files from a directory and sorts them by slice location."""
    slices = [pydicom.dcmread(p) for p in glob.glob(os.path.join(path, "*.dcm"))]

    # Sort by ImagePositionPatient Z coordinate (Robust Fallback)
    def get_z(s):
        if hasattr(s, "ImagePositionPatient"):
            return float(s.ImagePositionPatient[2])
        if hasattr(s, "SliceLocation"):
            return float(s.SliceLocation)
        if hasattr(s, "InstanceNumber"):
            return float(s.InstanceNumber)
        return 0.0

    slices.sort(key=get_z)

    # Check for slice thickness consistency and infer if missing
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except (AttributeError, IndexError):
        try:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        except (AttributeError, IndexError):
            slice_thickness = getattr(slices[0], "SliceThickness", 1.0)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    """Converts raw DICOM pixel_array to Hounsfield Units."""
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def generate_tri_slabs(volume, img_size=240):
    """
    Generates Fixed Overlapping Orthogonal Tri-Slabs (Axial and Coronal).
    Returns normalized RGB images (H, W, 3).
    """
    # 1. Normalize Volume (Lung Window: W=1500, L=-500 -> [-1250, 250])
    # Using a slightly wider range to capture more tissue info: [-1000, 400]
    min_hu, max_hu = -1000, 400
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)  # Normalize to [0, 1]

    # 2. Define Slicing Logic for 3 overlapping slabs
    def get_slabs_mip(vol_axis_data):
        """
        Input: Volume data oriented such that the axis to split is axis 0.
        Shape: (Depth, H, W)
        """
        depth = vol_axis_data.shape[0]
        if depth < 3:
            # Handle edge case with very few slices by repeating
            return np.stack([np.max(vol_axis_data, axis=0)] * 3, axis=-1)

        # Define split points
        # 3 slabs, 15% overlap
        # Slab 1: 0% to 33% + 15%
        # Slab 2: 33% - 15% to 66% + 15%
        # Slab 3: 66% - 15% to 100%

        p1 = int(depth * 0.33)
        p2 = int(depth * 0.66)
        overlap = int(depth * 0.15)

        # Indices
        s1_start, s1_end = 0, min(depth, p1 + overlap)
        s2_start, s2_end = max(0, p1 - overlap), min(depth, p2 + overlap)
        s3_start, s3_end = max(0, p2 - overlap), depth

        # Compute MIPs
        mip1 = np.max(vol_axis_data[s1_start:s1_end], axis=0)
        mip2 = np.max(vol_axis_data[s2_start:s2_end], axis=0)
        mip3 = np.max(vol_axis_data[s3_start:s3_end], axis=0)

        # Stack to RGB
        img = np.stack([mip1, mip2, mip3], axis=-1)
        return img

    # Axial View (Split along Z / axis 0)
    axial_img = get_slabs_mip(volume)

    # Coronal View (Split along Y / axis 1)
    # Transpose to make Y the first axis: (Z, Y, X) -> (Y, Z, X)
    coronal_vol = np.transpose(volume, (1, 0, 2))
    coronal_img = get_slabs_mip(coronal_vol)

    # Resize to target size
    axial_img = cv2.resize(axial_img, (img_size, img_size))
    coronal_img = cv2.resize(coronal_img, (img_size, img_size))

    return axial_img, coronal_img


def get_patient_images(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Retrieves patient images from cache or processes them from DICOMs.
    Returns: axial_img, coronal_img (numpy arrays)
    """
    os.makedirs(cache_dir, exist_ok=True)

    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from Scratch
    # Construct full path to DICOMs. dicom_dir is relative (e.g., "train/ID...")
    full_dicom_path = os.path.join(Config.INPUT_DIR, dicom_dir)

    try:
        scans = load_scan(full_dicom_path)
        volume = get_pixels_hu(scans)
        axial, coronal = generate_tri_slabs(volume, Config.IMG_SIZE)

        # 3. Save to Cache
        try:
            np.save(axial_path, axial)
            np.save(coronal_path, coronal)
        except Exception:
            pass  # Ignore write errors (e.g. race conditions)

        return axial, coronal

    except Exception as e:
        # Fallback for empty/corrupt directories: return black images
        print(f"Error processing {patient_id}: {e}")
        empty = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Save fallback to cache to prevent repeated failures (Cite debug_lesson_8)
        try:
            np.save(axial_path, empty)
            np.save(coronal_path, empty)
        except Exception:
            pass

        return empty, empty


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class OSICDataset(Dataset):
    def __init__(
        self, df, cache_dir, transform=None, mode="train", load_cached_data=True
    ):
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Preprocess Metadata
        self._prepare_metadata()

    def _prepare_metadata(self):
        """
        Augments dataframe with Baseline features and encodes categorical variables.
        """
        # 1. Identify Baseline Features
        if self.mode in ["train", "val"]:
            # Group by Patient to find baseline (min Weeks)
            # We assume the input DF has multiple rows per patient

            # Create a baseline lookup
            # Sort by Weeks to ensure first is baseline
            sorted_df = self.df.sort_values(["Patient", "Weeks"])
            baseline_df = sorted_df.groupby("Patient").first().reset_index()

            # Columns to merge back
            cols = ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus", "Weeks"]
            baseline_df = baseline_df[cols]

            # Rename for merge
            rename_map = {
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
                "Weeks": "Baseline_Week",
            }
            baseline_df = baseline_df.rename(columns=rename_map)

            # Merge back to main df
            self.df = pd.merge(self.df, baseline_df, on="Patient", how="left")

            # Calculate Delta Week (Current - Baseline)
            # In train.csv, 'Weeks' is already relative to baseline CT (Week 0).
            # However, the baseline FVC measurement might be at Week != 0.
            self.df["Delta_Week"] = self.df["Weeks"] - self.df["Baseline_Week"]

        elif self.mode == "test":
            # Test CSV already has Baseline_* columns and Predict_Week
            self.df["Delta_Week"] = self.df["Predict_Week"] - self.df["Baseline_Week"]

        # 2. Encode Features
        # Sex: Male=0, Female=1
        self.df["Sex_Enc"] = self.df["Baseline_Sex"].map({"Male": 0, "Female": 1})

        # Smoking: One-Hot-ish (Mapped to integers for embedding or direct use)
        # We will create 3 columns for One-Hot manually to ensure order
        # Categories: 'Ex-smoker', 'Never smoked', 'Currently smokes'
        self.df["Smoke_Ex"] = (self.df["Baseline_SmokingStatus"] == "Ex-smoker").astype(
            float
        )
        self.df["Smoke_Never"] = (
            self.df["Baseline_SmokingStatus"] == "Never smoked"
        ).astype(float)
        self.df["Smoke_Current"] = (
            self.df["Baseline_SmokingStatus"] == "Currently smokes"
        ).astype(float)

        # Normalize Numerical Features
        # Simple scaling based on typical ranges
        self.df["Age_Scaled"] = self.df["Baseline_Age"] / 100.0
        self.df["Percent_Scaled"] = self.df["Baseline_Percent"] / 100.0
        self.df["Base_FVC_Scaled"] = self.df["Baseline_FVC"] / 1000.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        dicom_dir = row["dicom_dir"]
        axial, coronal = get_patient_images(
            patient_id, dicom_dir, self.cache_dir, self.load_cached_data
        )

        # 2. Apply Transforms
        if self.transform:
            # Albumentations requires key 'image'
            res_ax = self.transform(image=axial)["image"]
            res_cor = self.transform(image=coronal)["image"]
        else:
            # Manual ToTensor if no transform provided
            res_ax = torch.from_numpy(axial.transpose(2, 0, 1))
            res_cor = torch.from_numpy(coronal.transpose(2, 0, 1))

        # 3. Prepare Tabular Vector
        # [Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Percent, Base_FVC]
        tabular = np.array(
            [
                row["Age_Scaled"],
                row["Sex_Enc"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
                row["Percent_Scaled"],
                row["Base_FVC_Scaled"],
            ],
            dtype=np.float32,
        )

        # 4. Prepare Metadata & Target
        meta = {
            "Patient_Week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{row['Weeks']}"
            ),
            "Delta_Week": float(row["Delta_Week"]),
            "Baseline_FVC": float(row["Baseline_FVC"]),
        }

        if self.mode in ["train", "val"]:
            target = float(row["FVC"])
            return {
                "axial": res_ax,
                "coronal": res_cor,
                "tabular": torch.tensor(tabular),
                "target": torch.tensor(target, dtype=torch.float32),
                "meta": meta,
            }
        else:
            return {
                "axial": res_ax,
                "coronal": res_cor,
                "tabular": torch.tensor(tabular),
                "meta": meta,
            }


# -------------------------------------------------------------------------
# DataLoaders
# -------------------------------------------------------------------------


def get_dataloaders(config):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    # Define Transforms
    # Spatial Only for Train: Flips, ShiftScaleRotate. No Intensity changes.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_test_transform = A.Compose(
        [
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Load DataFrames
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Create Datasets
    # We enable caching. The first run will be slow, subsequent runs fast.
    train_dataset = OSICDataset(
        train_df,
        config.CACHE_DIR,
        transform=train_transform,
        mode="train",
        load_cached_data=True,
    )

    val_dataset = OSICDataset(
        val_df,
        config.CACHE_DIR,
        transform=val_test_transform,
        mode="val",
        load_cached_data=True,
    )

    test_dataset = OSICDataset(
        test_df,
        config.CACHE_DIR,
        transform=val_test_transform,
        mode="test",
        load_cached_data=True,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
