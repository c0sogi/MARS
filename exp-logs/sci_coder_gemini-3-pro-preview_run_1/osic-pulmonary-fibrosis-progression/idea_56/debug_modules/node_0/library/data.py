import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from library.config import Config

# Try importing pydicom, handle case if missing (though required for DICOM)
try:
    import pydicom
except ImportError:
    pydicom = None


def get_transforms(mode="train"):
    """
    Returns the Albumentations composition for the given mode.
    Strictly spatial augmentations (flips, shifts), no intensity changes.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            ]
        )
    else:
        return A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE)])


def load_dicom_volume(path):
    """
    Reads a directory of DICOM files, sorts them by Z-position,
    and converts to Hounsfield Units (HU).
    Returns a 3D numpy array (Depth, Height, Width).
    """
    if not os.path.exists(path):
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    slices = []
    for f in files:
        try:
            if pydicom:
                dcm = pydicom.dcmread(f)
                slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Sort slices by Z-position (ImagePositionPatient[2])
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback to InstanceNumber
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass  # Keep file order if all else fails

    # Stack and convert to HU
    images = []
    for s in slices:
        try:
            # Convert to float
            img = s.pixel_array.astype(np.float32)

            # Apply RescaleSlope and RescaleIntercept to get HU
            intercept = getattr(s, "RescaleIntercept", -1024.0)
            slope = getattr(s, "RescaleSlope", 1.0)

            if slope != 1:
                img = slope * img.astype(np.float64)
                img = img.astype(np.float32)

            img += intercept
            images.append(img)
        except Exception:
            continue

    if not images:
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    vol = np.stack(images)
    return vol


def generate_tri_slab(vol, view="axial"):
    """
    Generates the Fixed Overlapping Orthogonal Tri-Slab representation.
    Splits volume into 3 overlapping slabs (0-38%, 31-69%, 62-100%).
    Computes MIP for each slab.
    Stacks to RGB (3 channels).
    Resizes to 224x224.
    """
    # Handle Coronal View: Transpose (D, H, W) -> (H, D, W)
    # This treats the original Height (Y-axis) as the new Depth
    if view == "coronal":
        vol = np.transpose(vol, (1, 0, 2))

    D, H, W = vol.shape

    # Define Indices for 3 slabs with ~15% overlap
    # Heuristic: Slab 1 (0-38%), Slab 2 (31-69%), Slab 3 (62-100%)
    p1 = int(D * 0.38)
    p2_start = int(D * 0.31)
    p2_end = int(D * 0.69)
    p3_start = int(D * 0.62)

    # Ensure valid indices
    p1 = max(1, p1)
    p2_end = max(p2_start + 1, p2_end)
    p3_start = min(D - 1, p3_start)

    if D < 3:
        # Volume too shallow, duplicate
        slab1 = vol
        slab2 = vol
        slab3 = vol
    else:
        slab1 = vol[:p1]
        slab2 = vol[p2_start:p2_end]
        slab3 = vol[p3_start:]

    # Compute MIPs (Maximum Intensity Projection)
    # Handle empty slabs just in case
    if slab1.size == 0:
        slab1 = vol
    if slab2.size == 0:
        slab2 = vol
    if slab3.size == 0:
        slab3 = vol

    mip1 = np.max(slab1, axis=0)
    mip2 = np.max(slab2, axis=0)
    mip3 = np.max(slab3, axis=0)

    # Stack to (H, W, 3)
    img = np.stack([mip1, mip2, mip3], axis=-1)

    # Normalize
    # Clip to lung window/range [-1000, 600]
    img = np.clip(img, -1000, 600)

    # Min-Max Scale to [0, 1]
    img = (img - (-1000)) / (600 - (-1000))

    # Resize to Target Size (224, 224)
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    return img.astype(np.float32)


class LungDataset(Dataset):
    def __init__(self, csv_path, mode="train", transform=None, debug=False):
        self.mode = mode
        self.transform = transform
        self.df = pd.read_csv(csv_path)

        # Debugging: Subsample
        if debug:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)

        # Preprocess Metadata for Training
        # We need Baseline_FVC and Baseline_Week for every row to calculate DeltaWeek
        # In test.csv, these columns already exist. In train.csv, we must compute them.
        if "Baseline_FVC" not in self.df.columns:
            # Identify baseline for each patient (Week closest to 0)
            self.df["abs_week"] = self.df["Weeks"].abs()
            # Sort by patient and abs_week to get closest to 0 first
            sorted_df = self.df.sort_values(["Patient", "abs_week"])
            # Drop duplicates to keep first (closest to 0)
            baseline_df = sorted_df.drop_duplicates("Patient")

            # Create mapping
            baseline_map = baseline_df.set_index("Patient")[["FVC", "Weeks"]].to_dict(
                "index"
            )

            # Map back to main df
            self.df["Baseline_FVC"] = self.df["Patient"].map(
                lambda x: baseline_map[x]["FVC"]
            )
            self.df["Baseline_Week"] = self.df["Patient"].map(
                lambda x: baseline_map[x]["Weeks"]
            )

        # Mappings for categorical variables
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # ==========================
        # 1. Image Loading & Caching
        # ==========================
        cache_ax_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
        cache_cor_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

        # Try load cache
        img_ax, img_cor = None, None
        if os.path.exists(cache_ax_path) and os.path.exists(cache_cor_path):
            try:
                img_ax = np.load(cache_ax_path)
                img_cor = np.load(cache_cor_path)
            except Exception:
                pass  # Fallback to processing

        if img_ax is None:
            img_ax, img_cor = self._process_and_cache(
                row, cache_ax_path, cache_cor_path
            )

        # ==========================
        # 2. Augmentations
        # ==========================
        # Albumentations works on (H, W, C)
        if self.transform:
            img_ax = self.transform(image=img_ax)["image"]
            img_cor = self.transform(image=img_cor)["image"]

        # Convert to Channel-First (C, H, W) for PyTorch
        img_ax = np.transpose(img_ax, (2, 0, 1))
        img_cor = np.transpose(img_cor, (2, 0, 1))

        # ==========================
        # 3. Tabular & Target
        # ==========================
        # Feature Vector: [Percent, Age, Sex, Smoking]
        # Simple scaling for numerical stability
        meta_vec = np.array(
            [
                row["Percent"] / 100.0,
                row["Age"] / 100.0,
                self.sex_map.get(row["Sex"], 0),
                self.smoke_map.get(row["SmokingStatus"], 0),
            ],
            dtype=np.float32,
        )

        # Aux Info
        delta_week = row["Weeks"] - row["Baseline_Week"]
        baseline_fvc = row["Baseline_FVC"]
        true_fvc = row["FVC"]

        return {
            "img_ax": torch.tensor(img_ax, dtype=torch.float32),
            "img_cor": torch.tensor(img_cor, dtype=torch.float32),
            "meta": torch.tensor(meta_vec, dtype=torch.float32),
            "delta_week": torch.tensor(delta_week, dtype=torch.float32),
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "fvc": torch.tensor(true_fvc, dtype=torch.float32),
            "patient_id": patient_id,
            "week": torch.tensor(row["Weeks"], dtype=torch.float32),
        }

    def _process_and_cache(self, row, cache_ax_path, cache_cor_path):
        # Construct path: Config.INPUT_ROOT + relative path from metadata
        full_path = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])

        vol = load_dicom_volume(full_path)

        img_ax = generate_tri_slab(vol, view="axial")
        img_cor = generate_tri_slab(vol, view="coronal")

        # Save cache
        np.save(cache_ax_path, img_ax)
        np.save(cache_cor_path, img_cor)

        return img_ax, img_cor
