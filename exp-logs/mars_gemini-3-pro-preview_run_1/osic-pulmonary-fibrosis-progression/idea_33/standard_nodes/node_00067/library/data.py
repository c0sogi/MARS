import os
import cv2
import torch
import numpy as np
import pandas as pd
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import seed_everything


def read_dicom_scan(dicom_dir):
    """
    Reads a DICOM series from a directory, sorts by InstanceNumber,
    and converts to a Hounsfield Unit (HU) numpy volume.
    """
    if not os.path.exists(dicom_dir):
        # Fallback for missing directories (should not happen based on metadata check)
        return np.zeros((10, 224, 224), dtype=np.float32)

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, 224, 224), dtype=np.float32)

    # Read all dicom files
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return np.zeros((10, 224, 224), dtype=np.float32)

    # Sort by InstanceNumber
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        # Fallback if InstanceNumber is missing, sort by filename
        pass

    # Extract images and convert to HU
    images = []
    for s in slices:
        try:
            img = s.pixel_array.astype(np.float32)
        except RuntimeError:
            continue

        # Convert to HU
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            img = slope * img.astype(np.float64)
            img = img.astype(np.float32)

        img += np.float32(intercept)
        images.append(img)

    if not images:
        return np.zeros((10, 224, 224), dtype=np.float32)

    # Stack to (D, H, W)
    volume = np.stack(images, axis=0)
    return volume


def generate_tri_slab_views(volume):
    """
    Generates Axial and Coronal Tri-Slab RGB images from a 3D volume.

    Args:
        volume (np.ndarray): 3D array (D, H, W) in Hounsfield Units.

    Returns:
        axial_img (np.ndarray): (224, 224, 3) normalized [0, 1]
        coronal_img (np.ndarray): (224, 224, 3) normalized [0, 1]
    """
    # 1. Normalize Volume
    # Clip to Lung Window [-1000, 400]
    volume = np.clip(volume, -1000, 400)
    # Normalize to [0, 1]
    volume = (volume - (-1000)) / (400 - (-1000))

    D, H, W = volume.shape

    # Helper to create slabs
    def create_slabs(vol_axis, axis_len):
        # Define slab boundaries with overlap
        # We want 3 slabs covering the axis_len
        # Slab size approx axis_len / 3
        # Overlap is 15% of slab size? Or just fixed overlap.
        # Let's use the logic: 3 intervals.
        # If we simply divide by 3:
        p1 = axis_len // 3
        p2 = 2 * (axis_len // 3)

        # Add overlap
        overlap = int(axis_len * Config.SLAB_OVERLAP)

        # Slab 1: 0 to p1 + overlap
        s1 = vol_axis[0 : min(axis_len, p1 + overlap), ...]
        # Slab 2: p1 - overlap to p2 + overlap
        s2 = vol_axis[max(0, p1 - overlap) : min(axis_len, p2 + overlap), ...]
        # Slab 3: p2 - overlap to end
        s3 = vol_axis[max(0, p2 - overlap) : axis_len, ...]

        # Handle empty slabs (small volumes)
        if s1.size == 0:
            s1 = np.zeros_like(vol_axis[0:1])
        if s2.size == 0:
            s2 = s1
        if s3.size == 0:
            s3 = s1

        return s1, s2, s3

    # --- Axial View (Axis 0 is Depth) ---
    a1, a2, a3 = create_slabs(volume, D)

    # MIP
    mip_a1 = np.max(a1, axis=0)
    mip_a2 = np.max(a2, axis=0)
    mip_a3 = np.max(a3, axis=0)

    # Stack channels
    axial_img = np.stack([mip_a1, mip_a2, mip_a3], axis=-1)  # (H, W, 3)

    # --- Coronal View (Axis 1 is Height/Y) ---
    # Transpose volume to make Axis 1 the primary depth for slicing
    vol_cor = volume.transpose(1, 0, 2)  # (H, D, W)
    c1, c2, c3 = create_slabs(vol_cor, H)

    mip_c1 = np.max(c1, axis=0)  # (D, W)
    mip_c2 = np.max(c2, axis=0)
    mip_c3 = np.max(c3, axis=0)

    coronal_img = np.stack([mip_c1, mip_c2, mip_c3], axis=-1)  # (D, W, 3)

    # --- Resize ---
    # cv2.resize expects (W, H)
    axial_img = cv2.resize(axial_img, (Config.IMG_SIZE, Config.IMG_SIZE))
    coronal_img = cv2.resize(coronal_img, (Config.IMG_SIZE, Config.IMG_SIZE))

    return axial_img, coronal_img


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Spatial only: Flips, Shifts, Rotations.
    No brightness/contrast changes to preserve HU density signal.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet stats
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


class LungDataset(Dataset):
    def __init__(self, csv_path, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            csv_path (str): Path to metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Load Metadata
        self.df = pd.read_csv(csv_path)

        # Debugging subset
        if Config.DEBUG_SAMPLE_SIZE is not None:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)

        # Preprocess Tabular Data
        self._preprocess_tabular()

        # Ensure cache dir exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def _preprocess_tabular(self):
        """
        Derives Baseline values and normalizes features.
        """
        # 1. Identify Baseline for Train/Val if not present
        if "Baseline_FVC" not in self.df.columns:
            # Group by Patient to find the row with min Weeks
            # We create a mapping Patient -> Baseline Data
            baseline_df = self.df.loc[self.df.groupby("Patient")["Weeks"].idxmin()]
            baseline_map = baseline_df.set_index("Patient")[
                ["FVC", "Percent", "Weeks"]
            ].to_dict("index")

            # Map back to main df
            self.df["Baseline_FVC"] = self.df["Patient"].apply(
                lambda x: baseline_map[x]["FVC"]
            )
            self.df["Baseline_Percent"] = self.df["Patient"].apply(
                lambda x: baseline_map[x]["Percent"]
            )
            self.df["Baseline_Week"] = self.df["Patient"].apply(
                lambda x: baseline_map[x]["Weeks"]
            )

            # Calculate Delta Week (Current - Baseline)
            self.df["Delta_Week"] = self.df["Weeks"] - self.df["Baseline_Week"]
        else:
            # Test set already has Baseline columns
            # Calculate Delta Week (Predict_Week - Baseline_Week)
            self.df["Delta_Week"] = self.df["Predict_Week"] - self.df["Baseline_Week"]

        # 2. Normalize/Encode Features
        # Hardcoded stats from EDA to ensure consistency
        # Age: Mean ~67, Std ~7
        self.df["Age_norm"] = (
            (self.df["Age"] - 67.0) / 7.0
            if "Age" in self.df.columns
            else (self.df["Baseline_Age"] - 67.0) / 7.0
        )

        # Percent: Mean ~77, Std ~19 (We use Baseline Percent for the input context)
        self.df["BasePercent_norm"] = (self.df["Baseline_Percent"] - 77.0) / 19.0

        # Sex: Male=0, Female=1
        sex_col = "Sex" if "Sex" in self.df.columns else "Baseline_Sex"
        self.df["Sex_enc"] = self.df[sex_col].map({"Male": 0, "Female": 1})

        # Smoking: One-Hot Encoding manually to keep vector size fixed
        # Categories: Ex-smoker, Never smoked, Currently smokes
        smoke_col = (
            "SmokingStatus"
            if "SmokingStatus" in self.df.columns
            else "Baseline_SmokingStatus"
        )
        self.df["Smoke_Ex"] = (self.df[smoke_col] == "Ex-smoker").astype(float)
        self.df["Smoke_Never"] = (self.df[smoke_col] == "Never smoked").astype(float)
        self.df["Smoke_Current"] = (self.df[smoke_col] == "Currently smokes").astype(
            float
        )

    def _get_images(self, patient_id, dicom_rel_path):
        """
        Retrieves images from cache or generates them from DICOM.
        """
        cache_axial = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
        cache_coronal = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

        # Try loading from cache
        if (
            self.load_cached_data
            and os.path.exists(cache_axial)
            and os.path.exists(cache_coronal)
        ):
            try:
                axial = np.load(cache_axial)
                coronal = np.load(cache_coronal)
                return axial, coronal
            except Exception:
                pass  # Fallback to generation

        # Generate from scratch
        full_dicom_path = os.path.join(Config.INPUT_ROOT, dicom_rel_path)
        volume = read_dicom_scan(full_dicom_path)
        axial, coronal = generate_tri_slab_views(volume)

        # Save to cache
        np.save(cache_axial, axial)
        np.save(cache_coronal, coronal)

        return axial, coronal

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Get Images
        img_axial, img_coronal = self._get_images(patient_id, row["dicom_dir"])

        # 2. Apply Transforms
        if self.transform:
            # Albumentations expects uint8 or float32. Our images are float32 [0,1].
            # We need to apply the same transform to both or independently?
            # Independent spatial transforms might break correspondence, but since they are orthogonal views,
            # they are already spatially distinct. Independent augmentation is fine and adds regularization.
            res_ax = self.transform(image=img_axial)
            img_axial = res_ax["image"]

            res_cor = self.transform(image=img_coronal)
            img_coronal = res_cor["image"]
        else:
            # ToTensor manually if no transform provided
            img_axial = torch.tensor(img_axial.transpose(2, 0, 1), dtype=torch.float32)
            img_coronal = torch.tensor(
                img_coronal.transpose(2, 0, 1), dtype=torch.float32
            )

        # 3. Tabular Features (Context for GLU)
        # Vector: [Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, BasePercent]
        tab_vec = np.array(
            [
                row["Age_norm"],
                row["Sex_enc"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
                row["BasePercent_norm"],
            ],
            dtype=np.float32,
        )

        # 4. Meta Features (Anchor for Head)
        # Vector: [Baseline_FVC, Delta_Week]
        # Note: Baseline_FVC is typically large (~2000-5000).
        # The model head will use this as a bias or scaling factor.
        # We pass it raw as requested by the "Prior-Anchored" design.
        meta_vec = np.array([row["Baseline_FVC"], row["Delta_Week"]], dtype=np.float32)

        # 5. Target
        if self.mode != "test":
            target = torch.tensor([row["FVC"]], dtype=torch.float32)
        else:
            target = torch.tensor([0.0], dtype=torch.float32)  # Dummy

        return {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "meta": torch.tensor(meta_vec, dtype=torch.float32),
            "target": target,
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{row['Weeks']}"
            ),
        }
