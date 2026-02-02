import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Try importing pydicom; handle environment where it might be missing strictly
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config

# ==========================================
# Helper Functions: DICOM Processing
# ==========================================


def get_img_coords(s):
    """Helper to sort DICOM slices by ImagePositionPatient Z-coordinate."""
    try:
        return float(s.ImagePositionPatient[2])
    except:
        return float(s.InstanceNumber)


def process_dicom(dicom_dir):
    """
    Reads DICOM files, converts to HU, and generates Axial and Coronal Tri-Slab MIPs.
    Returns:
        img_axial: (224, 224, 3) uint8
        img_coronal: (224, 224, 3) uint8
    """
    if pydicom is None:
        # Fallback for environments without pydicom (should not happen based on task type)
        return np.zeros((224, 224, 3), dtype=np.uint8), np.zeros(
            (224, 224, 3), dtype=np.uint8
        )

    files = [
        os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")
    ]
    if not files:
        return np.zeros((224, 224, 3), dtype=np.uint8), np.zeros(
            (224, 224, 3), dtype=np.uint8
        )

    # Read and sort slices
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except:
            continue

    if not slices:
        return np.zeros((224, 224, 3), dtype=np.uint8), np.zeros(
            (224, 224, 3), dtype=np.uint8
        )

    slices.sort(key=get_img_coords)

    # Create 3D Volume
    # Handle Rescale Intercept/Slope
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)
        img = img * slope + intercept
        images.append(img)

    volume = np.stack(images)  # (D, H, W)

    # Lung Windowing [-1000, 400]
    volume = np.clip(volume, -1000, 400)
    # Normalize to [0, 1]
    volume = (volume + 1000) / 1400.0

    # Helper for Tri-Slab MIP
    def generate_tri_slab(vol_data, axis):
        # vol_data shape: (D, H, W) or (H, D, W) depending on view
        depth = vol_data.shape[0]
        if depth < 1:
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)

        # Define slab boundaries with overlap
        # Slabs: 0-33%, 33-66%, 66-100%
        # Overlap: 15% of the slab size
        slab_size = depth / 3.0
        overlap = slab_size * 0.15

        starts = [0, slab_size - overlap, 2 * slab_size - overlap]
        ends = [slab_size + overlap, 2 * slab_size + overlap, depth]

        channels = []
        for s, e in zip(starts, ends):
            s_idx = max(0, int(s))
            e_idx = min(depth, int(e))
            if s_idx >= e_idx:
                slab_mip = np.zeros_like(vol_data[0])
            else:
                # Compute MIP along the depth axis
                slab_mip = np.max(vol_data[s_idx:e_idx], axis=0)
            channels.append(slab_mip)

        img = np.stack(channels, axis=-1)  # (H, W, 3)

        # Resize to target resolution
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Convert to uint8 [0, 255]
        img = (img * 255).astype(np.uint8)
        return img

    # Axial View (Standard Z-axis)
    img_axial = generate_tri_slab(volume, axis=0)

    # Coronal View (Reslice along Y-axis)
    # Volume is (D, H, W). Coronal view looks from Front (Y).
    # We permute to (H, D, W) so Y becomes the depth dimension for the helper
    vol_coronal = np.transpose(volume, (1, 0, 2))
    img_coronal = generate_tri_slab(vol_coronal, axis=0)

    return img_axial, img_coronal


def get_images(patient_id, rel_dicom_dir, load_cached_data=True):
    """
    Retrieves images from cache or processes them from scratch.
    """
    cache_path_ax = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
    cache_path_cor = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_path_ax)
        and os.path.exists(cache_path_cor)
    ):
        try:
            img_ax = np.load(cache_path_ax)
            img_cor = np.load(cache_path_cor)
            return img_ax, img_cor
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    full_dicom_path = os.path.join(Config.INPUT_DIR, rel_dicom_dir)
    img_ax, img_cor = process_dicom(full_dicom_path)

    # 3. Save to cache
    np.save(cache_path_ax, img_ax)
    np.save(cache_path_cor, img_cor)

    return img_ax, img_cor


# ==========================================
# Dataset Class
# ==========================================


class LungDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Pre-process Tabular Data
        # We need to establish a "Baseline" for every patient in Train/Val
        if self.mode in ["train", "val"]:
            self.baseline_lookup = self._build_baseline_lookup(self.df)

    def _build_baseline_lookup(self, df):
        """
        Identifies the baseline visit (closest to Week 0) for each patient.
        Returns a dict: {PatientID: Series}
        """
        lookup = {}
        patient_groups = df.groupby("Patient")
        for pid, group in patient_groups:
            # Find row with min absolute weeks
            group["abs_weeks"] = group["Weeks"].abs()
            baseline_row = group.loc[group["abs_weeks"].idxmin()]
            lookup[pid] = baseline_row
        return lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Determine Baseline Features and Delta Week
        if self.mode in ["train", "val"]:
            base_data = self.baseline_lookup[pid]

            # Features from Baseline
            age = base_data["Age"]
            sex = base_data["Sex"]
            smoking = base_data["SmokingStatus"]
            percent = base_data["Percent"]
            base_fvc = base_data["FVC"]
            base_week = base_data["Weeks"]

            # Target info
            current_week = row["Weeks"]
            target_fvc = row["FVC"]
            dicom_dir = base_data["dicom_dir"]  # Should be same for all rows of patient

        else:  # Test mode
            # Test CSV structure is different (merged in metadata generation)
            # Columns: Baseline_Age, Baseline_Sex, etc.
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoking = row["Baseline_SmokingStatus"]
            percent = row["Baseline_Percent"]
            base_fvc = row["Baseline_FVC"]
            base_week = row["Baseline_Week"]

            current_week = row["Predict_Week"]
            target_fvc = 0  # Dummy
            dicom_dir = row["dicom_dir"]

        delta_week = current_week - base_week

        # 2. Construct Tabular Vector
        # [Age_norm, Sex_bin, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_norm]

        # Normalize Age: (Age - 65) / 15
        age_norm = (age - 65.0) / 15.0

        # Encode Sex: Male=0, Female=1
        sex_bin = 1.0 if sex == "Female" else 0.0

        # Encode Smoking: One-hot
        # Categories: 'Ex-smoker', 'Never smoked', 'Currently smokes'
        s_ex = 1.0 if smoking == "Ex-smoker" else 0.0
        s_never = 1.0 if smoking == "Never smoked" else 0.0
        s_curr = 1.0 if smoking == "Currently smokes" else 0.0

        # Normalize Percent: val / 100
        percent_norm = percent / 100.0

        tabular = np.array(
            [age_norm, sex_bin, s_ex, s_never, s_curr, percent_norm], dtype=np.float32
        )

        # 3. Load Images
        img_ax, img_cor = get_images(pid, dicom_dir, load_cached_data=True)

        # 4. Apply Transforms
        if self.transforms:
            # Albumentations expects dict for multiple images if using 'additional_targets'
            # But here we can just apply separately or use a composed transform that handles one
            # We apply the same spatial transform to both?
            # No, they are different views. Independent augmentation is fine or just spatial on Axial.
            # Let's apply transforms independently.
            res_ax = self.transforms(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transforms(image=img_cor)
            img_cor = res_cor["image"]
        else:
            # Just to tensor
            t = ToTensorV2()
            img_ax = t(image=img_ax)["image"]
            img_cor = t(image=img_cor)["image"]

        # Normalize if not done in transforms (Albumentations Normalize usually does /255 and mean/std)
        # Assuming transforms include Normalize. If not, we should do it.
        # We will define transforms to include Normalize.

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "delta_week": torch.tensor([delta_week], dtype=torch.float32),
            "base_fvc": torch.tensor([base_fvc], dtype=torch.float32),
            "target": torch.tensor([target_fvc], dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{pid}_{current_week}"
            ),
        }


# ==========================================
# Transforms
# ==========================================


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
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


# ==========================================
# Data Loaders
# ==========================================


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        # test_df = test_df.head(20) # Keep test full for submission check usually

    # Create Datasets
    train_ds = LungDataset(train_df, transforms=get_transforms("train"), mode="train")
    val_ds = LungDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_ds = LungDataset(test_df, transforms=get_transforms("test"), mode="test")

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
