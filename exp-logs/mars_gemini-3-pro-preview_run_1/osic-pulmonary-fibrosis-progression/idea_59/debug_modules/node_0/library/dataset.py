import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.utils import get_device

# Handle pydicom import
try:
    import pydicom
except ImportError:
    pydicom = None

# Constants
IMG_SIZE = 224
CACHE_DIR = "./working/idea_59"


def get_transforms(mode="train"):
    """
    Returns albumentations transforms.
    Spatial augmentation only for training. No intensity augmentation.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
            ]
        )
    else:
        return A.Compose([])


def load_scan(path):
    """
    Loads a CT scan from a directory of DICOM files.
    Returns a 3D numpy array (Z, H, W) in Hounsfield Units.
    """
    if pydicom is None:
        # Fallback if pydicom is strictly unavailable
        return np.zeros((10, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    slices = []
    if not os.path.exists(path):
        return np.zeros((10, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for f in files:
        try:
            s = pydicom.dcmread(os.path.join(path, f))
            slices.append(s)
        except:
            continue

    if not slices:
        return np.zeros((10, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Sort by ImagePositionPatient Z, or InstanceNumber as fallback
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass

    # Process slices
    images = []
    for s in slices:
        try:
            # Convert to HU
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = img * slope + intercept

            # Resize immediately to save memory
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
        except:
            continue

    if not images:
        return np.zeros((10, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    return np.stack(images)


def get_tri_slab(vol, axis):
    """
    Generates a 3-channel image using Fixed Overlapping Orthogonal Tri-Slabs.
    axis=0: Axial (Split Z)
    axis=1: Coronal (Split Y)
    """
    # If generating Coronal view (axis 1), permute so the split axis is 0
    # Vol is (Z, Y, X). For Coronal, we want to split Y.
    # Transpose to (Y, Z, X)
    if axis == 1:
        vol = vol.transpose(1, 0, 2)

    depth = vol.shape[0]

    # Define Tri-Slab boundaries with ~15% overlap
    # Overlap is calculated relative to total depth
    overlap = int(max(1, depth * 0.15))

    # Split points
    p1 = int(depth / 3)
    p2 = int(2 * depth / 3)

    # Define slab indices
    s1_start, s1_end = 0, min(depth, p1 + overlap // 2)
    s2_start, s2_end = max(0, p1 - overlap // 2), min(depth, p2 + overlap // 2)
    s3_start, s3_end = max(0, p2 - overlap // 2), depth

    # Helper for MIP
    def get_mip(start, end):
        if start >= end:
            # Fallback for empty slab
            return np.max(vol, axis=0)
        slab = vol[start:end, :, :]
        if slab.shape[0] == 0:
            return np.max(vol, axis=0)
        return np.max(slab, axis=0)

    c1 = get_mip(s1_start, s1_end)
    c2 = get_mip(s2_start, s2_end)
    c3 = get_mip(s3_start, s3_end)

    # Stack to RGB (H, W, 3)
    img = np.stack([c1, c2, c3], axis=-1)

    # Windowing: Lung Window [-1000, 400]
    img = np.clip(img, -1000, 400)
    # Normalize to [0, 1]
    img = (img + 1000) / 1400.0
    # Convert to uint8 [0, 255]
    img = (img * 255).astype(np.uint8)

    # Resize to target size (224, 224)
    # Note: For Coronal view, the resulting image is (Z, X). Z might not be 224.
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


class LungDataset(Dataset):
    def __init__(
        self, metadata_path, mode="train", transform=None, load_cached_data=True
    ):
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Load metadata
        try:
            df = pd.read_csv(metadata_path)
        except FileNotFoundError:
            print(f"Warning: Metadata file {metadata_path} not found.")
            df = pd.DataFrame()

        self.data = []

        if not df.empty:
            if mode in ["train", "val"]:
                # Group by patient to find baseline
                for patient, group in df.groupby("Patient"):
                    # Baseline is the visit closest to Week 0
                    group = group.copy()
                    group["abs_week"] = group["Weeks"].abs()
                    base_idx = group["abs_week"].idxmin()
                    base_row = group.loc[base_idx]

                    base_vals = {
                        "FVC": float(base_row["FVC"]),
                        "Percent": float(base_row["Percent"]),
                        "Age": float(base_row["Age"]),
                        "Sex": base_row["Sex"],
                        "SmokingStatus": base_row["SmokingStatus"],
                        "Week": float(base_row["Weeks"]),
                    }
                    dicom_dir = base_row["dicom_dir"]

                    for _, row in group.iterrows():
                        self.data.append(
                            {
                                "Patient": patient,
                                "dicom_dir": dicom_dir,
                                "Weeks": float(row["Weeks"]),
                                "FVC": float(row["FVC"]),
                                "base": base_vals,
                            }
                        )

            elif mode == "test":
                # Test metadata already has baseline info merged
                for _, row in df.iterrows():
                    base_vals = {
                        "FVC": float(row["Baseline_FVC"]),
                        "Percent": float(row["Baseline_Percent"]),
                        "Age": float(row["Baseline_Age"]),
                        "Sex": row["Baseline_Sex"],
                        "SmokingStatus": row["Baseline_SmokingStatus"],
                        "Week": float(row["Baseline_Week"]),
                    }
                    self.data.append(
                        {
                            "Patient": row["Patient"],
                            "dicom_dir": row["dicom_dir"],
                            "Weeks": float(row["Predict_Week"]),
                            "FVC": 0.0,  # Dummy target
                            "base": base_vals,
                            "Patient_Week": row["Patient_Week"],
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Image Loading (with Caching)
        axial, coronal = self._get_images(item["Patient"], item["dicom_dir"])

        # 2. Augmentation
        if self.transform:
            # Apply independently to preserve view-specific features
            axial = self.transform(image=axial)["image"]
            coronal = self.transform(image=coronal)["image"]

        # 3. Normalization & Tensor Conversion
        # Normalize to [0, 1] and convert to (C, H, W)
        axial = torch.tensor(axial.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        coronal = torch.tensor(coronal.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        # 4. Tabular Data Processing
        base = item["base"]

        # Normalize features
        # Age: Mean ~65, Range 50-90
        age = (base["Age"] - 65.0) / 15.0
        # Percent: Mean ~80, Range 30-150
        pct = (base["Percent"] - 80.0) / 20.0
        # Sex: Male=0, Female=1
        sex = 0.0 if base["Sex"] == "Male" else 1.0
        # Smoking: One-hot encoding
        smk = [0.0, 0.0, 0.0]
        s_stat = base["SmokingStatus"]
        if s_stat == "Ex-smoker":
            smk[0] = 1.0
        elif s_stat == "Never smoked":
            smk[1] = 1.0
        elif s_stat == "Currently smokes":
            smk[2] = 1.0

        # Base FVC scaled (approx mean 2700)
        base_fvc_scaled = base["FVC"] / 3000.0

        # Construct Tabular Vector (7 dims)
        # [Age, Sex, Smk_Ex, Smk_Never, Smk_Curr, Percent, Base_FVC_Scaled]
        tab_vec = torch.tensor(
            [age, sex] + smk + [pct, base_fvc_scaled], dtype=torch.float32
        )

        # 5. Targets & Meta
        delta_week = item["Weeks"] - base["Week"]
        target_fvc = item["FVC"]
        base_fvc_raw = base["FVC"]

        return {
            "axial": axial,
            "coronal": coronal,
            "tabular": tab_vec,
            "delta_week": torch.tensor(delta_week, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc_raw, dtype=torch.float32),
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "patient_week": item.get("Patient_Week", ""),
        }

    def _get_images(self, patient_id, dicom_dir):
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        ax_path = os.path.join(CACHE_DIR, f"{patient_id}_axial.npy")
        cor_path = os.path.join(CACHE_DIR, f"{patient_id}_coronal.npy")

        # Try to load from cache
        if (
            self.load_cached_data
            and os.path.exists(ax_path)
            and os.path.exists(cor_path)
        ):
            try:
                return np.load(ax_path), np.load(cor_path)
            except:
                pass  # Fallback to recompute

        # Compute from scratch
        try:
            full_path = os.path.join("./input", dicom_dir)
            vol = load_scan(full_path)

            # Generate views
            axial = get_tri_slab(vol, axis=0)
            coronal = get_tri_slab(vol, axis=1)

            # Save to cache
            np.save(ax_path, axial)
            np.save(cor_path, coronal)

            return axial, coronal
        except Exception as e:
            # Return blank images on failure to prevent crash
            print(f"Error processing images for {patient_id}: {e}")
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            return img, img


def get_dataloaders(batch_size=16, num_workers=2):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    train_ds = LungDataset(
        "./metadata/train.csv", mode="train", transform=get_transforms("train")
    )
    val_ds = LungDataset(
        "./metadata/val.csv", mode="val", transform=get_transforms("val")
    )
    test_ds = LungDataset(
        "./metadata/test.csv", mode="test", transform=get_transforms("test")
    )

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
