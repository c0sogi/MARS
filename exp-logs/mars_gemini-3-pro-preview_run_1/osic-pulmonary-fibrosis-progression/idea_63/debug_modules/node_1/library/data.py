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
# Helper Functions for Image Processing
# ==========================================


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by slice location,
    and converts them to a 3D numpy array in Hounsfield Units (HU).
    """
    slices = []
    try:
        for s in os.listdir(path):
            if s.endswith(".dcm"):
                slices.append(pydicom.dcmread(os.path.join(path, s)))
    except Exception as e:
        # Fallback if directory listing fails or pydicom has issues
        print(f"Error loading scan from {path}: {e}")
        return None

    if not slices:
        return None

    # Sort slices by ImagePositionPatient Z-coordinate
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback: sort by InstanceNumber if position is missing
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Convert to Hounsfield Units (HU)
    image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Apply Intercept and Slope if present
    # Most CTs have these tags. If not, assume raw pixel data is close enough or handle error.
    if hasattr(slices[0], "RescaleIntercept") and hasattr(slices[0], "RescaleSlope"):
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.float32)

        image += np.float32(intercept)

    return image


def generate_tri_slab(volume, view="axial", target_size=224, overlap_ratio=0.15):
    """
    Generates a 3-channel image (RGB) by splitting the volume into 3 overlapping slabs
    along the depth axis and computing the Maximum Intensity Projection (MIP) for each.

    Args:
        volume (np.array): 3D array (Z, Y, X) in HU.
        view (str): 'axial' or 'coronal'.
        target_size (int): Output spatial resolution.
        overlap_ratio (float): Percentage of slab size to overlap.

    Returns:
        np.array: (target_size, target_size, 3) normalized image in [0, 1].
    """
    # 1. Pre-process Volume (Lung Windowing)
    # Clip to standard lung window [-1000, 400]
    volume = np.clip(volume, -1000, 400)

    # Normalize to [0, 1]
    volume = (volume - (-1000)) / (400 - (-1000))

    # 2. Orient Volume based on View
    if view == "coronal":
        # Axial is (Z, Y, X). Coronal view is looking from Front (Y).
        # We want to split along Y.
        # Transpose to (Y, Z, X) so the splitting logic is identical
        volume = np.transpose(volume, (1, 0, 2))
    # If axial, volume is already (Z, Y, X), we split along Z (dim 0).

    depth = volume.shape[0]

    # Handle edge case with very few slices
    if depth < 3:
        # Repeat volume to fill depth
        volume = np.concatenate([volume] * (4 // depth + 1), axis=0)
        depth = volume.shape[0]

    # 3. Define Slabs with Overlap
    slab_size = depth / 3.0
    overlap = slab_size * overlap_ratio

    # Indices
    idx1_start = 0
    idx1_end = int(slab_size + overlap)

    idx2_start = int(slab_size - overlap)
    idx2_end = int(2 * slab_size + overlap)

    idx3_start = int(2 * slab_size - overlap)
    idx3_end = depth

    # Clamp indices
    idx1_end = min(idx1_end, depth)
    idx2_start = max(0, idx2_start)
    idx2_end = min(idx2_end, depth)
    idx3_start = max(0, idx3_start)

    # 4. Compute MIPs
    # MIP along the depth axis (axis 0)
    # If slice is empty, use zeros
    def get_mip(start, end):
        if start >= end:
            return np.zeros((volume.shape[1], volume.shape[2]), dtype=np.float32)
        slab = volume[start:end, :, :]
        return np.max(slab, axis=0)

    c1 = get_mip(idx1_start, idx1_end)  # Red
    c2 = get_mip(idx2_start, idx2_end)  # Green
    c3 = get_mip(idx3_start, idx3_end)  # Blue

    # 5. Stack and Resize
    # Stack to (H, W, 3)
    img = np.stack([c1, c2, c3], axis=-1)

    # Resize to target resolution
    # Note: For Coronal, the aspect ratio might be weird (Z vs X).
    # We force resize to square as per BBSL-Net requirements.
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return img


# ==========================================
# Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(self, mode="train", transform=None, debug=False):
        """
        Args:
            mode (str): 'train', 'val', or 'submission'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, use a small subset.
        """
        self.mode = mode
        self.transform = transform
        self.debug = debug
        self.input_root = Config.input_root
        self.cache_dir = Config.cache_dir

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.train_csv)
            self._prepare_train_metadata()
        elif mode == "val":
            self.df = pd.read_csv(Config.val_csv)
            self._prepare_train_metadata()
        elif mode == "submission":
            self.df = pd.read_csv(Config.test_csv)
            # Test CSV from metadata generation already has Baseline columns
            # Rename columns to match internal schema if necessary
            # Schema: Patient, Predict_Week, Baseline_Week, Baseline_FVC, ...
            # We map Predict_Week -> Weeks for consistency
            if "Predict_Week" in self.df.columns:
                self.df = self.df.rename(columns={"Predict_Week": "Weeks"})

        if self.debug:
            self.df = self.df.head(Config.debug_sample_size)

        # Pre-compute unique patients for caching checks
        self.patients = self.df["Patient"].unique()

    def _prepare_train_metadata(self):
        """
        For training/validation, we need to identify the baseline visit
        to provide static context (Age, Percent, FVC at baseline)
        and calculate the time delta.
        """
        # Group by Patient to find baseline
        # Baseline is defined as the visit with the minimum Weeks value
        baseline_df = self.df.loc[self.df.groupby("Patient")["Weeks"].idxmin()]

        # Select relevant baseline features
        baseline_df = baseline_df[
            ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
        ]
        baseline_df = baseline_df.rename(
            columns={
                "Weeks": "Baseline_Week",
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
            }
        )

        # Merge baseline features back to the main dataframe
        # Note: We drop the original static columns from main df to avoid confusion,
        # except 'Weeks' and 'FVC' which are the targets for the specific visit.
        self.df = self.df[["Patient", "Weeks", "FVC"]]
        self.df = self.df.merge(baseline_df, on="Patient", how="left")

    def _get_image(self, patient_id, view):
        """
        Retrieves the processed image from cache or generates it from DICOMs.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}_{view}.npy")

        # 1. Try Load from Cache
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path).astype(np.float32)
            except:
                pass  # File corrupted, regenerate

        # 2. Generate from Scratch
        # We need to load the volume first.
        # Optimization: To avoid loading volume twice (for axial and coronal),
        # we could cache the volume, but memory is tight.
        # We will load DICOMs, process both views, and save both to cache at once.

        dicom_dir = os.path.join(
            self.input_root,
            "train" if self.mode != "submission" else "test",
            patient_id,
        )
        if not os.path.exists(dicom_dir):
            # Fallback for test set structure if different
            dicom_dir = os.path.join(self.input_root, "test", patient_id)

        volume = load_scan(dicom_dir)

        if volume is None:
            # Return black image if data is missing
            return np.zeros((Config.image_size, Config.image_size, 3), dtype=np.float32)

        # Generate both views
        img_axial = generate_tri_slab(
            volume, "axial", Config.image_size, Config.slab_overlap
        )
        img_coronal = generate_tri_slab(
            volume, "coronal", Config.image_size, Config.slab_overlap
        )

        # Save both to cache
        np.save(os.path.join(self.cache_dir, f"{patient_id}_axial.npy"), img_axial)
        np.save(os.path.join(self.cache_dir, f"{patient_id}_coronal.npy"), img_coronal)

        if view == "axial":
            return img_axial
        else:
            return img_coronal

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial & Coronal)
        img_ax = self._get_image(patient_id, "axial")
        img_cor = self._get_image(patient_id, "coronal")

        # 2. Apply Augmentations
        if self.transform:
            # Apply same spatial transform to both?
            # Usually we want independent or consistent?
            # For "Spatial Only" like flips, it's better to be consistent if they were registered,
            # but here they are orthogonal views. Independent augmentation is fine and adds robustness.
            res_ax = self.transform(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor = res_cor["image"]
        else:
            # Just to tensor
            t = ToTensorV2()
            img_ax = t(image=img_ax)["image"]
            img_cor = t(image=img_cor)["image"]

        # 3. Process Tabular Data
        # Normalize continuous features
        # Age: (Age - 50) / 50
        age_norm = (row["Baseline_Age"] - 50.0) / 50.0

        # Percent: (Percent - 75) / 25 (Approx mean 75, std 20)
        # Or simple / 100
        percent_norm = row["Baseline_Percent"] / 100.0

        # Categorical Encoding
        # Sex: Male=0, Female=1
        sex_feat = 1.0 if row["Baseline_Sex"] == "Female" else 0.0

        # Smoking: Never=0, Ex=1, Currently=2 (Arbitrary mapping, MLP learns embedding)
        smk_status = row["Baseline_SmokingStatus"]
        if smk_status == "Never smoked":
            smk_feat = 0.0
        elif smk_status == "Ex-smoker":
            smk_feat = 1.0
        else:  # Currently smokes
            smk_feat = 2.0

        # Tabular Vector: [Age, Sex, Smoking, Percent]
        tabular = torch.tensor(
            [age_norm, sex_feat, smk_feat, percent_norm], dtype=torch.float32
        )

        # 4. Time Delta
        # The model predicts parameters based on static data.
        # The loss function uses the time delta to project the prediction.
        # Delta = Current_Week - Baseline_Week
        time_delta = float(row["Weeks"] - row["Baseline_Week"])

        # 5. Target
        # For submission, FVC might be dummy, but we need Baseline_FVC for the anchor
        target_fvc = float(row["FVC"])
        baseline_fvc = float(row["Baseline_FVC"])

        return {
            "img_ax": img_ax,  # (3, 224, 224)
            "img_cor": img_cor,  # (3, 224, 224)
            "tabular": tabular,  # (4,)
            "time_delta": time_delta,  # Scalar
            "baseline_fvc": baseline_fvc,  # Scalar
            "target": target_fvc,  # Scalar
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{row['Weeks']}"
            ),
        }


# ==========================================
# Transforms
# ==========================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Strictly spatial-only for training to preserve HU density semantics.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Resize(Config.image_size, Config.image_size),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(Config.image_size, Config.image_size), ToTensorV2()])


# ==========================================
# Data Loaders
# ==========================================


def get_dataloaders():
    """
    Creates DataLoaders for Train, Validation, and Submission.
    """
    # Train Loader
    train_dataset = OSICDataset(
        mode="train", transform=get_transforms("train"), debug=Config.debug
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Loader
    val_dataset = OSICDataset(
        mode="val", transform=get_transforms("val"), debug=Config.debug
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Submission Loader (Test Set)
    sub_dataset = OSICDataset(
        mode="submission", transform=get_transforms("val"), debug=Config.debug
    )
    sub_loader = DataLoader(
        sub_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, sub_loader
