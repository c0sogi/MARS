import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_dicom_volume(dicom_dir):
    """
    Reads .dcm files from a directory, sorts them by instance number or position,
    converts to Hounsfield Units, and returns a 3D numpy array (D, H, W).
    """
    if not os.path.exists(dicom_dir):
        return None

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return None

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, f))
            # Verify pixel data exists
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except:
            continue

    if not slices:
        return None

    # Sort slices
    # Primary: InstanceNumber, Secondary: ImagePositionPatient Z
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            pass  # Fallback to filesystem order

    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)
        img = slope * img + intercept
        images.append(img)

    if not images:
        return None

    return np.stack(images)


def generate_tri_slab(volume, axis_idx):
    """
    Generates a 3-channel Tri-Slab MIP image.
    axis_idx: 0 for Axial (Z-axis), 1 for Coronal (Y-axis).
    Returns a (H, W, 3) uint8 image resized to Config.IMG_SIZE.
    """
    # If Coronal, transpose (D, H, W) -> (H, D, W) so we can slice the first dimension
    if axis_idx == 1:
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # Define 3 overlapping slabs
    # Slab 1: 0% - 40%
    # Slab 2: 30% - 70%
    # Slab 3: 60% - 100%
    p1_end = int(depth * 0.40)
    p2_start = int(depth * 0.30)
    p2_end = int(depth * 0.70)
    p3_start = int(depth * 0.60)

    # Boundary checks
    p1_end = max(1, p1_end)
    p2_start = max(0, min(p2_start, depth - 1))
    p2_end = max(p2_start + 1, p2_end)
    p3_start = max(0, min(p3_start, depth - 1))

    s1 = volume[0:p1_end, :, :]
    s2 = volume[p2_start:p2_end, :, :]
    s3 = volume[p3_start:, :, :]

    # Handle edge cases (single slice volumes)
    if s1.shape[0] == 0:
        s1 = volume
    if s2.shape[0] == 0:
        s2 = volume
    if s3.shape[0] == 0:
        s3 = volume

    # Maximum Intensity Projection
    c1 = np.max(s1, axis=0)
    c2 = np.max(s2, axis=0)
    c3 = np.max(s3, axis=0)

    img = np.stack([c1, c2, c3], axis=-1)  # (H, W, 3)

    # Lung Windowing: Level -600, Width 1500 -> Range [-1350, 150]
    min_hu = -1350.0
    max_hu = 150.0

    img = np.clip(img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)  # Normalize to 0-1

    # Resize to model input size
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    # Convert to uint8 for storage efficiency
    img = (img * 255).astype(np.uint8)

    return img


def get_transforms(phase):
    """
    Returns Albumentations transforms.
    Strictly spatial augmentations only (no brightness/contrast).
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()])


class OSICDataset(Dataset):
    def __init__(self, csv_path, mode="train", transform=None):
        self.mode = mode
        self.transform = transform if transform else get_transforms(mode)
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Raw Metadata
        df = pd.read_csv(csv_path)

        # Process Metadata into standardized format
        self.data = self._process_dataframe(df)

        # Debugging subset
        if Config.DEBUG_SAMPLE_SIZE and len(self.data) > Config.DEBUG_SAMPLE_SIZE:
            self.data = self.data.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(
                drop=True
            )

    def _process_dataframe(self, df):
        """
        Standardizes the input dataframe.
        - For Train/Val: Groups by patient history to find baseline stats and compute Delta_Week.
        - For Test: Uses provided baseline columns and computes Delta_Week.
        """
        processed_rows = []

        # Check if it's the test set format (contains Baseline_FVC)
        if "Baseline_FVC" in df.columns:
            # Test Set Logic
            for _, row in df.iterrows():
                # Encode Categoricals
                sex = 0 if row["Baseline_Sex"] == "Male" else 1

                smk = 1  # Never
                if row["Baseline_SmokingStatus"] == "Ex-smoker":
                    smk = 0
                elif row["Baseline_SmokingStatus"] == "Currently smokes":
                    smk = 2

                # Normalize Continuous
                age_norm = (row["Baseline_Age"] - 50) / 20.0
                pct_norm = row["Baseline_Percent"] / 100.0

                processed_rows.append(
                    {
                        "Patient": row["Patient"],
                        "Patient_Week": row["Patient_Week"],
                        "dicom_dir": row["dicom_dir"],
                        "Target_FVC": 0,  # Dummy for inference
                        "Delta_Week": row["Predict_Week"] - row["Baseline_Week"],
                        "Baseline_FVC": row["Baseline_FVC"],
                        "Baseline_Percent": row["Baseline_Percent"],
                        "Age_Norm": age_norm,
                        "Percent_Norm": pct_norm,
                        "Sex_Enc": sex,
                        "Smoke_Enc": smk,
                    }
                )
        else:
            # Train/Val Set Logic (Full History)
            # Group by Patient to identify baseline (min weeks)
            for pid, group in df.groupby("Patient"):
                group = group.sort_values("Weeks")
                baseline = group.iloc[0]

                base_fvc = baseline["FVC"]
                base_pct = baseline["Percent"]
                base_week = baseline["Weeks"]

                # Static features from baseline visit
                sex = 0 if baseline["Sex"] == "Male" else 1
                smk = 1
                if baseline["SmokingStatus"] == "Ex-smoker":
                    smk = 0
                elif baseline["SmokingStatus"] == "Currently smokes":
                    smk = 2

                age_norm = (baseline["Age"] - 50) / 20.0
                pct_norm = base_pct / 100.0

                dicom_dir = baseline["dicom_dir"]

                # Create a sample for every visit in the history
                for _, row in group.iterrows():
                    processed_rows.append(
                        {
                            "Patient": pid,
                            "Patient_Week": f"{pid}_{row['Weeks']}",
                            "dicom_dir": dicom_dir,
                            "Target_FVC": row["FVC"],
                            "Delta_Week": row["Weeks"] - base_week,
                            "Baseline_FVC": base_fvc,
                            "Baseline_Percent": base_pct,
                            "Age_Norm": age_norm,
                            "Percent_Norm": pct_norm,
                            "Sex_Enc": sex,
                            "Smoke_Enc": smk,
                        }
                    )

        return pd.DataFrame(processed_rows)

    def get_images_for_patient(self, patient_id, dicom_rel_path, load_cached_data=True):
        """
        Retrieves Axial and Coronal images.
        Checks cache first; if missing, processes DICOMs and saves to cache.
        """
        cache_ax = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        cache_cor = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_ax) and os.path.exists(cache_cor):
            try:
                return np.load(cache_ax), np.load(cache_cor)
            except:
                pass  # Corrupt file, reprocess

        # 2. Process from Scratch
        full_path = os.path.join(Config.INPUT_DIR, dicom_rel_path)
        vol = load_dicom_volume(full_path)

        if vol is None:
            # Fallback: Black image
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            return img, img

        img_ax = generate_tri_slab(vol, 0)
        img_cor = generate_tri_slab(vol, 1)

        # 3. Save Cache
        try:
            np.save(cache_ax, img_ax)
            np.save(cache_cor, img_cor)
        except:
            pass

        return img_ax, img_cor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Load Images
        img_ax, img_cor = self.get_images_for_patient(row["Patient"], row["dicom_dir"])

        # 2. Augmentations
        if self.transform:
            # Apply independently to views
            img_ax = self.transform(image=img_ax)["image"]
            img_cor = self.transform(image=img_cor)["image"]

        # 3. Tabular Features
        # Vector for GLU (Context): [Age_Norm, Percent_Norm, Sex_OH(2), Smoke_OH(3)] -> Dim 7
        sex_oh = np.zeros(2, dtype=np.float32)
        sex_oh[int(row["Sex_Enc"])] = 1.0

        smk_oh = np.zeros(3, dtype=np.float32)
        smk_oh[int(row["Smoke_Enc"])] = 1.0

        tab_glu = np.concatenate(
            [
                np.array([row["Age_Norm"], row["Percent_Norm"]], dtype=np.float32),
                sex_oh,
                smk_oh,
            ]
        )

        # Vector for Skip Connection (Priors): [Baseline_FVC_Scaled, Baseline_Percent_Scaled, Age_Norm]
        # Scaling FVC by 1000 to keep range reasonable (~2.0 - 5.0)
        tab_skip = np.array(
            [
                row["Baseline_FVC"] / 1000.0,
                row["Baseline_Percent"] / 100.0,
                row["Age_Norm"],
            ],
            dtype=np.float32,
        )

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tab_glu": torch.tensor(tab_glu, dtype=torch.float32),
            "tab_skip": torch.tensor(tab_skip, dtype=torch.float32),
            "delta_week": torch.tensor(row["Delta_Week"], dtype=torch.float32),
            "target": torch.tensor(row["Target_FVC"], dtype=torch.float32),
            "patient_week": row["Patient_Week"],
            "baseline_fvc": torch.tensor(row["Baseline_FVC"], dtype=torch.float32),
        }
