import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom
from library.config import Config

# ==========================================
# Constants & Encoders
# ==========================================
SEX_MAP = {"Male": 0, "Female": 1}
SMOKE_MAP = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train' for augmentation, 'val'/'test' for deterministic formatting.
    """
    if mode == "train":
        return A.Compose(
            [
                # Spatial Augmentations only
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                # Normalization and Tensor Conversion
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


def generate_tri_slab(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Generates Fixed Overlapping Orthogonal Tri-Slab MIPs (Axial and Coronal).

    Args:
        patient_id (str): Unique Patient ID.
        dicom_dir (str): Path to the directory containing DICOM files.
        cache_dir (str): Directory to save/load .npy files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (axial_mip, coronal_mip) as numpy arrays (224, 224, 3).
    """
    ax_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    cor_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(ax_path) and os.path.exists(cor_path):
        try:
            return np.load(ax_path), np.load(cor_path)
        except Exception:
            pass  # Fallback to regeneration if load fails

    def save_and_return_dummy():
        dummy = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        np.save(ax_path, dummy)
        np.save(cor_path, dummy)
        return dummy, dummy

    # 2. Load DICOMs
    if not os.path.exists(dicom_dir):
        # Fallback for missing directories (should not happen in valid dataset)
        return save_and_return_dummy()

    files = [
        os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")
    ]
    if not files:
        return save_and_return_dummy()

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            # Ensure ImagePositionPatient exists
            if hasattr(dcm, "ImagePositionPatient"):
                slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return save_and_return_dummy()

    # Sort by Z position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 3. Convert to HU and Windowing
    # Standard Lung Window: Level -600, Width 1500 -> Range [-1350, 150]
    L, W = -600, 1500
    lower, upper = L - W // 2, L + W // 2

    img_list = []
    for s in slices:
        try:
            # Convert to HU
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", 0)
            pix = s.pixel_array.astype(np.float32) * slope + intercept

            # Clip and Normalize
            pix = np.clip(pix, lower, upper)
            pix = (pix - lower) / (upper - lower)  # [0, 1]

            # Resize slice immediately to reduce memory usage for 3D volume
            # Original slices are often 512x512
            pix_resized = cv2.resize(pix, (Config.IMG_SIZE, Config.IMG_SIZE))
            img_list.append(pix_resized)
        except Exception:
            continue

    if not img_list:
        return save_and_return_dummy()

    volume = np.stack(img_list)  # (Depth, 224, 224)

    # Handle small volumes
    if volume.shape[0] < 3:
        volume = np.concatenate([volume] * 3, axis=0)

    # 4. Helper for Tri-Slab MIP
    def get_mips(vol, axis_size):
        # Overlapping splits: 0-40%, 30-70%, 60-100%
        # This provides roughly 15% overlap relative to a 33% split
        s1 = vol[: int(axis_size * 0.4)]
        s2 = vol[int(axis_size * 0.3) : int(axis_size * 0.7)]
        s3 = vol[int(axis_size * 0.6) :]

        # Handle edge case if slice count is very small and indices overlap weirdly
        if s1.shape[0] == 0:
            s1 = vol
        if s2.shape[0] == 0:
            s2 = vol
        if s3.shape[0] == 0:
            s3 = vol

        m1 = np.max(s1, axis=0)
        m2 = np.max(s2, axis=0)
        m3 = np.max(s3, axis=0)

        return np.stack([m1, m2, m3], axis=-1)  # (H, W, 3)

    # 5. Axial Processing
    # Volume is already (Depth, H, W) -> Axial view
    axial_mip = get_mips(volume, volume.shape[0])
    axial_mip = (axial_mip * 255).astype(np.uint8)

    # 6. Coronal Processing
    # Transpose to (Y, Depth, X) -> (H, D, W)
    # In medical imaging, Coronal is usually X-Z plane viewed from Y.
    # Here we treat the second dimension (Height in Axial) as the depth for Coronal.
    vol_cor = volume.transpose(1, 0, 2)  # (H, D, W)
    coronal_mip = get_mips(vol_cor, vol_cor.shape[0])

    # Resize Coronal to square (it is currently D x W or similar aspect ratio)
    coronal_mip = cv2.resize(coronal_mip, (Config.IMG_SIZE, Config.IMG_SIZE))
    coronal_mip = (coronal_mip * 255).astype(np.uint8)

    # 7. Save to Cache
    np.save(ax_path, axial_mip)
    np.save(cor_path, coronal_mip)

    return axial_mip, coronal_mip


class LungDataset(Dataset):
    def __init__(self, df, cache_dir, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            cache_dir (str): Directory for cached images.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        # Pre-calculate Baseline Features for Train/Val
        # In Train/Val, we have multiple rows per patient. We must use the Baseline (Week ~0)
        # clinical features as input to predict the trajectory, to match inference.
        if self.mode in ["train", "val"]:
            self.baseline_lookup = {}
            # Group by patient
            for pid, group in self.df.groupby("Patient"):
                # Find row closest to Week 0
                group["abs_weeks"] = group["Weeks"].abs()
                baseline_row = group.sort_values("abs_weeks").iloc[0]

                self.baseline_lookup[pid] = {
                    "Age": baseline_row["Age"],
                    "Sex": baseline_row["Sex"],
                    "SmokingStatus": baseline_row["SmokingStatus"],
                    "Percent": baseline_row["Percent"],
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Load Images
        # We assume data is prepared via prepare_data, but fallback just in case
        try:
            axial = np.load(os.path.join(self.cache_dir, f"{pid}_axial.npy"))
            coronal = np.load(os.path.join(self.cache_dir, f"{pid}_coronal.npy"))
        except FileNotFoundError:
            dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
            axial, coronal = generate_tri_slab(
                pid, dicom_dir, self.cache_dir, load_cached_data=False
            )

        # 2. Apply Transforms
        if self.transform:
            # Apply independent augmentations to views
            aug_ax = self.transform(image=axial)["image"]
            aug_cor = self.transform(image=coronal)["image"]
        else:
            t = ToTensorV2()
            aug_ax = t(image=axial)["image"]
            aug_cor = t(image=coronal)["image"]

        # 3. Prepare Tabular Features
        # Input Vector: [Age_norm, Sex_M, Sex_F, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_norm]

        if self.mode in ["train", "val"]:
            feats = self.baseline_lookup[pid]
            age_raw = feats["Age"]
            sex_raw = feats["Sex"]
            smoke_raw = feats["SmokingStatus"]
            pct_raw = feats["Percent"]
        else:
            # In Test mode, columns are prefixed with Baseline_
            age_raw = row["Baseline_Age"]
            sex_raw = row["Baseline_Sex"]
            smoke_raw = row["Baseline_SmokingStatus"]
            pct_raw = row["Baseline_Percent"]

        # Normalize Numerical
        # Age: scaled approx (Age - 65) / 15
        age_norm = (age_raw - 65.0) / 15.0
        # Percent: scaled approx (Percent - 80) / 20
        pct_norm = (pct_raw - 80.0) / 20.0

        # One-Hot Categorical
        sex_vec = np.zeros(2, dtype=np.float32)
        sex_idx = SEX_MAP.get(sex_raw, 0)  # Default to 0 if unknown
        sex_vec[sex_idx] = 1.0

        smoke_vec = np.zeros(3, dtype=np.float32)
        smoke_idx = SMOKE_MAP.get(smoke_raw, 1)  # Default to 'Never' (1) if unknown
        smoke_vec[smoke_idx] = 1.0

        # Concatenate: 1 + 2 + 3 + 1 = 7 dimensions
        tab_vec = np.concatenate([[age_norm], sex_vec, smoke_vec, [pct_norm]]).astype(
            np.float32
        )

        # 4. Prepare Output
        data = {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "metadata": {
                "Patient": pid,
                "Weeks": row["Predict_Week"] if "Predict_Week" in row else row["Weeks"],
                "Baseline_Week": (
                    row["Baseline_Week"] if "Baseline_Week" in row else 0
                ),  # Assuming 0 for train
                "Patient_Week": (
                    row["Patient_Week"]
                    if "Patient_Week" in row
                    else f"{pid}_{row['Weeks']}"
                ),
            },
        }

        # Add target for training
        if self.mode != "test":
            data["target"] = torch.tensor(row["FVC"], dtype=torch.float32)
            # For training, Baseline_Week is effectively 0 relative to the baseline scan
            data["metadata"]["Baseline_Week"] = 0

            # We also need the Baseline FVC for the loss/metric calculation if we were doing residual prediction
            # But the model predicts parameters, and we calculate loss against True FVC.
            # We might need Baseline FVC to calculate the 'delta' in the head if we used a residual approach.
            # However, based on the description, the head predicts alpha, sigma.
            # FVC_pred = Baseline_FVC + alpha * dt.
            # So we need Baseline_FVC in the batch to compute the prediction.

            # Let's add Baseline FVC to metadata for the model to use
            # In train mode, we need to find it similar to other feats
            # We can just look up the FVC at baseline week.
            # Re-using the baseline lookup logic:
            # Note: We didn't store FVC in baseline_lookup above. Let's assume the model needs it.
            # Actually, for training, we can pass it.

            # Find baseline FVC
            # (In a real scenario we'd optimize this look up)
            # For now, let's trust the model handles the math or we pass it here.
            # The prompt's model description: "Pass inputs ... through Network ... Obtain alpha ... Calculate predictions using anchored trajectory logic".
            # This implies the Network output is just parameters. The generic training loop handles the math.
            # The training loop needs Baseline FVC.

            # Let's add Baseline_FVC to metadata
            # We can get it from the dataframe if we merged it, but we didn't.
            # Let's fetch it from the same row we got the baseline features from.
            # (This is slightly inefficient inside getitem but safe)
            # Optimized: We already have baseline_lookup. Let's add FVC to it in __init__?
            # No, I'll just leave it. The training loop can infer it or we assume the input 'FVC' is the target.
            # Wait, the prediction equation requires Baseline_FVC.
            # I will add 'Baseline_FVC' to the metadata dict.

            # Hack for efficiency: We need to know the FVC at week ~0.
            # In __init__, I will add FVC to baseline_lookup.
            pass

        # Add Baseline FVC to metadata for Train/Val
        if self.mode in ["train", "val"]:
            data["metadata"]["Baseline_FVC"] = self.baseline_lookup[pid].get(
                "FVC_Base", 2000.0
            )

        return data

    def _extend_baseline_lookup(self):
        # Helper to ensure FVC is in lookup (called in init)
        pass


# Monkey-patching __init__ to include FVC in lookup for Train/Val
# I will rewrite the __init__ logic in the class above to be cleaner.
# (Self-correction applied in the final code block below)


def prepare_data(df, cache_dir, load_cached_data=True):
    """
    Pre-generates images for all patients in the dataframe.
    """
    unique_patients = df[["Patient", "dicom_dir"]].drop_duplicates()
    print(f"Preparing data for {len(unique_patients)} patients...")

    # Ensure cache dir exists
    os.makedirs(cache_dir, exist_ok=True)

    for _, row in unique_patients.iterrows():
        dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
        generate_tri_slab(row["Patient"], dicom_dir, cache_dir, load_cached_data)


# Redefining LungDataset to include FVC in baseline lookup cleanly
class LungDataset(Dataset):
    def __init__(self, df, cache_dir, transform=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        if self.mode in ["train", "val"]:
            self.baseline_lookup = {}
            for pid, group in self.df.groupby("Patient"):
                group["abs_weeks"] = group["Weeks"].abs()
                baseline_row = group.sort_values("abs_weeks").iloc[0]
                self.baseline_lookup[pid] = {
                    "Age": baseline_row["Age"],
                    "Sex": baseline_row["Sex"],
                    "SmokingStatus": baseline_row["SmokingStatus"],
                    "Percent": baseline_row["Percent"],
                    "FVC_Base": baseline_row["FVC"],
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        try:
            axial = np.load(os.path.join(self.cache_dir, f"{pid}_axial.npy"))
            coronal = np.load(os.path.join(self.cache_dir, f"{pid}_coronal.npy"))
        except FileNotFoundError:
            dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
            axial, coronal = generate_tri_slab(
                pid, dicom_dir, self.cache_dir, load_cached_data=False
            )

        if self.transform:
            aug_ax = self.transform(image=axial)["image"]
            aug_cor = self.transform(image=coronal)["image"]
        else:
            t = ToTensorV2()
            aug_ax = t(image=axial)["image"]
            aug_cor = t(image=coronal)["image"]

        if self.mode in ["train", "val"]:
            feats = self.baseline_lookup[pid]
            age_raw = feats["Age"]
            sex_raw = feats["Sex"]
            smoke_raw = feats["SmokingStatus"]
            pct_raw = feats["Percent"]
            base_fvc = feats["FVC_Base"]
        else:
            age_raw = row["Baseline_Age"]
            sex_raw = row["Baseline_Sex"]
            smoke_raw = row["Baseline_SmokingStatus"]
            pct_raw = row["Baseline_Percent"]
            base_fvc = row["Baseline_FVC"]

        age_norm = (age_raw - 65.0) / 15.0
        pct_norm = (pct_raw - 80.0) / 20.0

        sex_vec = np.zeros(2, dtype=np.float32)
        sex_vec[SEX_MAP.get(sex_raw, 0)] = 1.0

        smoke_vec = np.zeros(3, dtype=np.float32)
        smoke_vec[SMOKE_MAP.get(smoke_raw, 1)] = 1.0

        tab_vec = np.concatenate([[age_norm], sex_vec, smoke_vec, [pct_norm]]).astype(
            np.float32
        )

        data = {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "metadata": {
                "Patient": pid,
                "Weeks": row["Predict_Week"] if "Predict_Week" in row else row["Weeks"],
                "Baseline_Week": row["Baseline_Week"] if "Baseline_Week" in row else 0,
                "Baseline_FVC": float(base_fvc),
                "Patient_Week": (
                    row["Patient_Week"]
                    if "Patient_Week" in row
                    else f"{pid}_{row['Weeks']}"
                ),
            },
        }

        if self.mode != "test":
            data["target"] = torch.tensor(row["FVC"], dtype=torch.float32)

        return data
