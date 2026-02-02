import os
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
import glob
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from library.config import Config
from library.utils import seed_everything

# Constants for Normalization
AGE_MEAN, AGE_DIV = 65.0, 20.0
PERCENT_MEAN, PERCENT_DIV = 80.0, 20.0
LUNG_WIN_LEVEL = -600
LUNG_WIN_WIDTH = 1500


def get_transforms(mode="train"):
    """
    Returns the Albumentations transforms for the specific mode.
    Strictly spatial augmentations only for training.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by InstanceNumber/Position.
    Returns a list of pydicom datasets.
    """
    if not os.path.exists(path):
        return []

    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return []

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except:
            continue

    # Sort by ImagePositionPatient Z if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).
    Handles slope and intercept.
    """
    arrays = []
    valid_slices = []

    for s in slices:
        try:
            # Access pixel_array to trigger decompression (Cite debug_lesson_2)
            arr = s.pixel_array.astype(np.float32)
            arrays.append(arr)
            valid_slices.append(s)
        except RuntimeError:
            continue

    if not arrays:
        return None

    image = np.stack(arrays)

    # Convert to HU
    for i, s in enumerate(valid_slices):
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.float64)

        image[i] += np.float64(intercept)

    return image


def process_volume(vol_hu):
    """
    Applies Lung Window and normalizes to [0, 255].
    Window: Level -600, Width 1500 => [-1350, 150]
    """
    min_hu = LUNG_WIN_LEVEL - LUNG_WIN_WIDTH // 2
    max_hu = LUNG_WIN_LEVEL + LUNG_WIN_WIDTH // 2

    vol_hu = np.clip(vol_hu, min_hu, max_hu)
    # Normalize to 0-1 then 0-255
    vol_hu = (vol_hu - min_hu) / (max_hu - min_hu)
    vol_hu = (vol_hu * 255).astype(np.uint8)
    return vol_hu


def generate_tri_slabs(vol):
    """
    Generates 3 overlapping slabs along the first dimension (depth)
    and computes Maximum Intensity Projection (MIP) for each.
    Returns: (H, W, 3) image.
    """
    D, H, W = vol.shape
    if D == 0:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    if D < 3:
        # Fallback for very shallow scans: Max over all
        mip = np.max(vol, axis=0)
        return np.stack([mip, mip, mip], axis=-1)

    # Define slab boundaries with overlap
    p1 = D // 3
    p2 = 2 * D // 3
    overlap_px = int(D * Config.SLAB_OVERLAP)

    # Slab 1: 0 to 33% + overlap
    s1 = vol[0 : p1 + overlap_px, :, :]
    # Slab 2: 33% - overlap to 66% + overlap
    s2 = vol[max(0, p1 - overlap_px) : p2 + overlap_px, :, :]
    # Slab 3: 66% - overlap to 100%
    s3 = vol[max(0, p2 - overlap_px) :, :, :]

    # Compute MIPs
    m1 = np.max(s1, axis=0) if s1.shape[0] > 0 else np.zeros((H, W), dtype=vol.dtype)
    m2 = np.max(s2, axis=0) if s2.shape[0] > 0 else np.zeros((H, W), dtype=vol.dtype)
    m3 = np.max(s3, axis=0) if s3.shape[0] > 0 else np.zeros((H, W), dtype=vol.dtype)

    return np.stack([m1, m2, m3], axis=-1)


def resize_image(img):
    """Resizes image to Config.IMG_SIZE x Config.IMG_SIZE."""
    return cv2.resize(
        img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )


class LungDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train", cache_dir=Config.CACHE_DIR):
        self.transforms = transforms
        self.mode = mode
        self.cache_dir = cache_dir

        # Preprocess DataFrame to ensure Baseline info is available
        self.df = self._prepare_dataframe(df, mode)

        # Pre-fit encoders (simple manual mapping for stability)
        self.smoking_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        self.sex_map = {"Male": 0, "Female": 1}

    def _prepare_dataframe(self, df, mode):
        """
        Ensures each row has Baseline_FVC, Baseline_Percent, Baseline_Age, Baseline_Week.
        Calculates Delta_Week.
        """
        df = df.copy()

        if mode in ["train", "val"]:
            # Group by Patient to find baseline (min Week)
            # We assume the CT scan corresponds to the baseline visit
            baseline_info = []

            # Use groupby for efficiency
            for patient, group in df.groupby("Patient"):
                # Find row with min weeks (closest to baseline)
                base_row = group.loc[group["Weeks"].idxmin()]

                baseline_info.append(
                    {
                        "Patient": patient,
                        "Baseline_Week": base_row["Weeks"],
                        "Baseline_FVC": base_row["FVC"],
                        "Baseline_Percent": base_row["Percent"],
                        "Baseline_Age": base_row["Age"],
                        "Baseline_Sex": base_row["Sex"],
                        "Baseline_SmokingStatus": base_row["SmokingStatus"],
                    }
                )

            base_df = pd.DataFrame(baseline_info)

            # Merge back
            df = pd.merge(df, base_df, on="Patient", how="left")

            # Calculate Delta
            df["Delta_Week"] = df["Weeks"] - df["Baseline_Week"]

        elif mode == "test":
            # Test metadata already has Baseline_* columns and Predict_Week
            # Calculate Delta
            df["Delta_Week"] = df["Predict_Week"] - df["Baseline_Week"]

        return df

    def __len__(self):
        return len(self.df)

    def _get_tabular_features(self, row):
        """
        Constructs the normalized tabular feature vector.
        Vector: [Age_Norm, Sex_Bin, Smoke_Ex, Smoke_Nev, Smoke_Cur, Percent_Norm]
        """
        # 1. Age Norm
        age_norm = (row["Baseline_Age"] - AGE_MEAN) / AGE_DIV

        # 2. Sex Binary
        sex_bin = self.sex_map.get(row["Baseline_Sex"], 0)

        # 3. Smoking One-Hot
        smoke_status = row["Baseline_SmokingStatus"]
        smoke_idx = self.smoking_map.get(smoke_status, 1)  # Default to Never if unknown
        smoke_ohe = [0, 0, 0]
        smoke_ohe[smoke_idx] = 1

        # 4. Percent Norm
        pct_norm = (row["Baseline_Percent"] - PERCENT_MEAN) / PERCENT_DIV

        # Combine: 1 + 1 + 3 + 1 = 6 dims
        # Note: Model config expects LATENT_DIM=128 input to MLP.
        # The MLP input layer will be sized to 6.
        return np.array([age_norm, sex_bin] + smoke_ohe + [pct_norm], dtype=np.float32)

    def _process_and_cache_images(self, patient_id, dicom_dir):
        """
        Loads DICOM, generates Axial/Coronal Tri-Slabs, Resizes, and Caches.
        """
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # If both exist, load and return
        if os.path.exists(axial_path) and os.path.exists(coronal_path):
            try:
                axial = np.load(axial_path)
                coronal = np.load(coronal_path)
                return axial, coronal
            except:
                pass  # Corrupt, re-process

        # Process
        full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
        slices = load_scan(full_path)

        if not slices:
            # Return black images if load fails
            empty = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            return empty, empty

        vol_hu = get_pixels_hu(slices)  # (D, H, W)

        if vol_hu is None:
            # Return black images if load fails
            empty = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            # Save failure to cache to prevent retrying (Cite debug_lesson_8)
            np.save(axial_path, empty)
            np.save(coronal_path, empty)
            return empty, empty

        vol_norm = process_volume(vol_hu)  # (D, H, W) uint8

        # 1. Axial View (Z-axis is dim 0)
        axial_mip = generate_tri_slabs(vol_norm)
        axial_img = resize_image(axial_mip)

        # 2. Coronal View (Y-axis is dim 1)
        # Transpose to (H, D, W) so Y becomes the new depth
        coronal_vol = vol_norm.transpose(1, 0, 2)
        coronal_mip = generate_tri_slabs(coronal_vol)
        coronal_img = resize_image(coronal_mip)

        # Save to cache (atomic-ish write)
        np.save(axial_path, axial_img)
        np.save(coronal_path, coronal_img)

        return axial_img, coronal_img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Cached or Processed)
        axial_img, coronal_img = self._process_and_cache_images(
            patient_id, row["dicom_dir"]
        )

        # 2. Apply Transforms
        if self.transforms:
            # Albumentations expects dict for dual inputs if using 'additional_targets'
            # But here we have two independent images. Apply same transform logic?
            # Usually separate transforms or same seed.
            # For simplicity and robustness, apply independent transforms or just standard norm for test.
            # Since backbones are independent, independent augmentation is fine/good.

            res_ax = self.transforms(image=axial_img)
            axial_tensor = res_ax["image"]

            res_cor = self.transforms(image=coronal_img)
            coronal_tensor = res_cor["image"]
        else:
            # Fallback to simple tensor conversion
            axial_tensor = (
                torch.from_numpy(axial_img.transpose(2, 0, 1)).float() / 255.0
            )
            coronal_tensor = (
                torch.from_numpy(coronal_img.transpose(2, 0, 1)).float() / 255.0
            )

        # 3. Tabular Features
        tab_vec = self._get_tabular_features(row)

        # 4. Target & Meta
        # For model input, we need Delta_Week and Baseline_FVC (for scaling/offset if used in model,
        # though the prompt says model predicts parameters, so calculation is outside).
        # Actually, the model needs Delta_Week if it was predicting FVC directly,
        # but here the model predicts alpha/sigma based on static features.
        # The LOSS function needs Delta_Week to compute the predicted FVC from alpha.

        sample = {
            "patient_id": patient_id,
            "axial": axial_tensor,
            "coronal": coronal_tensor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "delta_week": torch.tensor(row["Delta_Week"], dtype=torch.float32),
            "baseline_fvc": torch.tensor(row["Baseline_FVC"], dtype=torch.float32),
        }

        if self.mode in ["train", "val"]:
            sample["target"] = torch.tensor(row["FVC"], dtype=torch.float32)

        return sample


def get_dataloaders(debug=False):
    """
    Factory function to create dataloaders for train, val, and test.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Transforms
    train_tf = get_transforms(mode="train")
    valid_tf = get_transforms(mode="val")

    # Datasets
    train_ds = LungDataset(train_df, transforms=train_tf, mode="train")
    val_ds = LungDataset(val_df, transforms=valid_tf, mode="val")
    test_ds = LungDataset(test_df, transforms=valid_tf, mode="test")

    # Loaders
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
