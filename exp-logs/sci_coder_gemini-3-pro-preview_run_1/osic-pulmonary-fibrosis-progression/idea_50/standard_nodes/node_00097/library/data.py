import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Attempt to import pydicom for DICOM handling
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline.
    Strictly spatial augmentations only. No intensity changes.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_scan(path):
    """
    Loads CT scan from a directory.
    Returns a 3D numpy array (D, H, W) normalized to 0-255 range.
    """
    if not os.path.exists(path):
        return np.zeros((10, 224, 224), dtype=np.uint8)

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, 224, 224), dtype=np.uint8)

    volume = []

    if HAS_PYDICOM:
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, f))
                slices.append(ds)
            except:
                pass

        if slices:
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

            if len(images) == 0:
                return np.zeros((10, 224, 224), dtype=np.uint8)

            volume = np.array(images)
        else:
            # Fallback if pydicom fails to read files
            files.sort(
                key=lambda x: int(os.path.splitext(x)[0]) if x[:-4].isdigit() else x
            )
            vol_list = []
            for f in files:
                img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    vol_list.append(img)
            if not vol_list:
                return np.zeros((10, 224, 224), dtype=np.uint8)
            volume = np.array(vol_list, dtype=np.float32)
    else:
        # Fallback if pydicom is not installed
        files.sort(key=lambda x: int(os.path.splitext(x)[0]) if x[:-4].isdigit() else x)
        vol_list = []
        for f in files:
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                vol_list.append(img)
        if not vol_list:
            return np.zeros((10, 224, 224), dtype=np.uint8)
        volume = np.array(vol_list, dtype=np.float32)

    # Normalize to 0-255 using Lung Window
    # Window: Level -500, Width 1500 -> [-1250, 250]
    # Standard Lung Window often cited as [-1000, 400] or [-1000, 600]
    # We use [-1000, 400] to cover air to soft tissue/bone
    min_hu = -1000
    max_hu = 400
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    volume = (volume * 255).astype(np.uint8)

    return volume


def generate_tri_slab(volume, view="axial"):
    """
    Generates a 3-channel RGB image using Tri-Slab MIP.
    volume: (D, H, W) or (C, D, H, W)
    view: 'axial' or 'coronal'
    """
    # Ensure volume is (D, H, W)
    if volume.ndim == 4:
        volume = volume[0]  # Handle cases where channel dim might exist

    if view == "coronal":
        # Transpose to make Coronal axis the depth (0-th dim)
        # Axial is (Z, Y, X). Coronal is (Y, Z, X)
        # Note: Depending on loading, usually (D, H, W) = (Z, Y, X)
        volume = volume.transpose(1, 0, 2)

    depth = volume.shape[0]
    if depth == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Define slabs: 0-33%, 33-66%, 66-100% with 15% overlap
    overlap = int(depth * 0.15)
    chunk_size = depth // 3

    # Slab 1
    s1_start = 0
    s1_end = min(depth, chunk_size + overlap)

    # Slab 2
    s2_start = max(0, chunk_size - overlap // 2)
    s2_end = min(depth, 2 * chunk_size + overlap // 2)

    # Slab 3
    s3_start = max(0, 2 * chunk_size - overlap)
    s3_end = depth

    def get_mip(start, end):
        if start >= end:
            return (
                volume[start : start + 1].max(axis=0)
                if start < depth
                else np.zeros_like(volume[0])
            )
        slab = volume[start:end]
        if slab.shape[0] == 0:
            return np.zeros_like(volume[0])
        return np.max(slab, axis=0)

    c1 = get_mip(s1_start, s1_end)
    c2 = get_mip(s2_start, s2_end)
    c3 = get_mip(s3_start, s3_end)

    # Stack to RGB
    img = np.stack([c1, c2, c3], axis=-1)  # (H, W, 3)

    # Resize to 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    return img


class LungDataset(Dataset):
    def __init__(
        self,
        mode="train",
        cache_dir="./working/idea_50/",
        load_cached_data=True,
        transform=None,
        limit_size=None,
    ):
        self.mode = mode
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data
        self.transform = transform

        # Load metadata
        if mode == "train":
            self.df = pd.read_csv("./metadata/train.csv")
        elif mode == "val":
            self.df = pd.read_csv("./metadata/val.csv")
        elif mode == "test":
            self.df = pd.read_csv("./metadata/test.csv")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if limit_size:
            self.df = self.df.iloc[:limit_size]

        # Ensure cache dir exists
        os.makedirs(self.cache_dir, exist_ok=True)

        self.input_root = "./input"

        # Precompute Baseline FVC for train/val sets
        self.baseline_lookup = {}
        if mode in ["train", "val"]:
            # For each patient, find the FVC at the earliest week (Baseline)
            # Note: We use the full dataframe (before limit_size) to find true baseline if possible,
            # but here we only have the split metadata. We assume the split contains the baseline visit.
            # In standard OSIC dataset, baseline is usually Week 0.
            for patient_id, group in self.df.groupby("Patient"):
                # Sort by Weeks
                sorted_group = group.sort_values("Weeks")
                # Take the first entry as baseline
                base_fvc = sorted_group.iloc[0]["FVC"]
                self.baseline_lookup[patient_id] = base_fvc

    def __len__(self):
        return len(self.df)

    def _load_processed_image(self, patient_id, dicom_rel_path):
        """
        Handles caching logic.
        """
        cache_path_ax = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        cache_path_cor = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try to load cached data
        if (
            self.load_cached_data
            and os.path.exists(cache_path_ax)
            and os.path.exists(cache_path_cor)
        ):
            try:
                img_ax = np.load(cache_path_ax)
                img_cor = np.load(cache_path_cor)
                return img_ax, img_cor
            except Exception:
                pass  # Load failed, recompute

        # 2. Compute from scratch
        full_path = os.path.join(self.input_root, dicom_rel_path)
        volume = load_scan(full_path)

        img_ax = generate_tri_slab(volume, view="axial")
        img_cor = generate_tri_slab(volume, view="coronal")

        # 3. Save to cache
        np.save(cache_path_ax, img_ax)
        np.save(cache_path_cor, img_cor)

        return img_ax, img_cor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Images
        img_ax, img_cor = self._load_processed_image(patient_id, row["dicom_dir"])

        # Apply transforms
        if self.transform:
            res_ax = self.transform(image=img_ax)["image"]
            res_cor = self.transform(image=img_cor)["image"]
        else:
            # Default to tensor
            res_ax = torch.tensor(img_ax.transpose(2, 0, 1)).float() / 255.0
            res_cor = torch.tensor(img_cor.transpose(2, 0, 1)).float() / 255.0

        # Metadata
        # Sex: Male=0, Female=1
        sex_str = row.get("Sex", row.get("Baseline_Sex"))
        sex = 0.0 if sex_str == "Male" else 1.0

        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        smk_str = row.get("SmokingStatus", row.get("Baseline_SmokingStatus"))
        if smk_str == "Ex-smoker":
            smk = 0.0
        elif smk_str == "Never smoked":
            smk = 1.0
        else:
            smk = 2.0

        age = float(row.get("Age", row.get("Baseline_Age")))
        percent = float(row.get("Percent", row.get("Baseline_Percent")))

        # Shared Latent Input: [Age, Sex, Smoking, Percent]
        meta = torch.tensor([age, sex, smk, percent], dtype=torch.float32)

        # Target and Week
        if self.mode in ["train", "val"]:
            target = torch.tensor(row["FVC"], dtype=torch.float32)
            week = torch.tensor(row["Weeks"], dtype=torch.float32)
            base_fvc = self.baseline_lookup.get(patient_id, 2000.0)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)  # Dummy
            week = torch.tensor(row["Predict_Week"], dtype=torch.float32)
            base_fvc = float(row["Baseline_FVC"])

        return {
            "image_axial": res_ax,
            "image_coronal": res_cor,
            "meta": meta,
            "target": target,
            "week": week,
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{int(week)}"
            ),
        }
