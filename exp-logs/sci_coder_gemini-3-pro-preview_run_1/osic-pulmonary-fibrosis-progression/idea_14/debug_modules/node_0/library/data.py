import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Attempt to import pydicom. If not available, we will handle it in the processor.
try:
    import pydicom
except ImportError:
    pydicom = None


class LungDataProcessor:
    """
    Handles the loading of DICOM files, generation of Fixed Overlapping Orthogonal Tri-Slabs,
    and caching of the processed images to disk.
    """

    def __init__(self, cache_dir="./working/idea_14"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_dicom_files(self, rel_dir):
        """
        Returns sorted list of DICOM files for a patient.
        rel_dir: Path relative to input root (e.g., 'train/ID000...')
        """
        full_path = os.path.join("./input", rel_dir)
        if not os.path.exists(full_path):
            return []

        files = [
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if f.endswith(".dcm")
        ]
        return files

    def read_dicom_volume(self, files):
        """
        Reads a list of DICOM files into a 3D numpy array (Depth, Height, Width).
        Converts to Hounsfield Units.
        """
        if not files:
            return None

        if pydicom is None:
            # Fallback if pydicom is missing (should not happen in correct env)
            print("Error: pydicom is not installed. Cannot process DICOM files.")
            return np.zeros((len(files), 512, 512), dtype=np.float32)

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                slices.append(dcm)
            except Exception as e:
                continue

        if not slices:
            return None

        # Sort slices by Z-position (ImagePositionPatient[2]) or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Convert to HU
        images = []
        for s in slices:
            # Pixel array
            img = s.pixel_array.astype(np.float32)

            # Rescale to HU
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = img * slope + intercept
            images.append(img)

        # Stack to (Depth, Height, Width)
        volume = np.stack(images)
        return volume

    def generate_tri_slab(self, volume, axis=0):
        """
        Generates a 3-channel RGB image using Fixed Overlapping Tri-Slabs.
        axis=0: Axial View (Split along Depth)
        axis=1: Coronal View (Split along Height/Anterior-Posterior)
        """
        # If Coronal, permute to put Height in the 0-th dimension
        if axis == 1:
            # Original: (D, H, W) -> Permute to (H, D, W)
            volume = np.transpose(volume, (1, 0, 2))

        D = volume.shape[0]
        if D == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # Define overlapping slabs
        # Slab 1: 0% - 40%
        # Slab 2: 30% - 70%
        # Slab 3: 60% - 100%
        # This ensures coverage and overlap (approx 15% of total depth overlap between neighbors)

        p0, p30, p40 = 0, int(D * 0.30), int(D * 0.40)
        p60, p70, p100 = int(D * 0.60), int(D * 0.70), D

        # Handle edge case where D is very small
        if D < 5:
            slab1 = volume[:]
            slab2 = volume[:]
            slab3 = volume[:]
        else:
            slab1 = volume[p0 : max(p40, p0 + 1)]
            slab2 = volume[p30 : max(p70, p30 + 1)]
            slab3 = volume[p60:p100]

        # Maximum Intensity Projection (MIP) along the slab depth
        def get_mip(slab):
            if slab.shape[0] == 0:
                return np.zeros(volume.shape[1:], dtype=np.float32)
            return np.max(slab, axis=0)

        ch1 = get_mip(slab1)
        ch2 = get_mip(slab2)
        ch3 = get_mip(slab3)

        # Stack to (H, W, 3)
        merged = np.stack([ch1, ch2, ch3], axis=-1)

        # Normalize HU: Window [-1000, 400]
        min_hu, max_hu = -1000.0, 400.0
        merged = np.clip(merged, min_hu, max_hu)
        merged = (merged - min_hu) / (max_hu - min_hu) * 255.0
        merged = merged.astype(np.uint8)

        # Resize to 224x224
        resized = cv2.resize(merged, (224, 224), interpolation=cv2.INTER_AREA)
        return resized

    def get_images(self, patient_id, dicom_dir, load_cached_data=True):
        """
        Retrieves Axial and Coronal images.
        Uses caching mechanism to avoid re-processing.
        """
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(axial_path)
            and os.path.exists(coronal_path)
        ):
            try:
                img_ax = np.load(axial_path)
                img_cor = np.load(coronal_path)
                return img_ax, img_cor
            except Exception:
                pass  # Fallback to re-compute

        # 2. Compute from scratch
        files = self.get_dicom_files(dicom_dir)
        volume = self.read_dicom_volume(files)

        if volume is None:
            # Fallback for empty/missing data
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            return dummy, dummy

        img_ax = self.generate_tri_slab(volume, axis=0)
        img_cor = self.generate_tri_slab(volume, axis=1)

        # 3. Save to cache
        np.save(axial_path, img_ax)
        np.save(coronal_path, img_cor)

        return img_ax, img_cor


def get_transforms(mode="train"):
    """
    Returns albumentations transforms.
    Strictly spatial augmentations for training. No intensity shifts.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
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
    def __init__(self, df, processor, transforms=None, mode="train"):
        """
        df: DataFrame containing patient metadata.
        processor: Instance of LungDataProcessor.
        mode: 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.transforms = transforms
        self.mode = mode

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Pre-compute baselines for training/validation data
        # (Test data already has baseline columns in the provided metadata)
        if self.mode != "test":
            self._precompute_baselines()

    def _precompute_baselines(self):
        """
        For training data, we need to identify the baseline FVC and Week for each patient.
        We assume the baseline is the visit with the minimum Week number.
        """
        self.baseline_lookup = {}
        self.baseline_week_lookup = {}

        # Group by Patient to find the initial visit
        for pid, group in self.df.groupby("Patient"):
            min_idx = group["Weeks"].idxmin()
            base_row = group.loc[min_idx]
            self.baseline_lookup[pid] = base_row["FVC"]
            self.baseline_week_lookup[pid] = base_row["Weeks"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        dicom_dir = row["dicom_dir"]
        img_ax, img_cor = self.processor.get_images(
            patient_id, dicom_dir, load_cached_data=True
        )

        # 2. Apply Transforms
        if self.transforms:
            # Apply independently as they are different views
            res_ax = self.transforms(image=img_ax)
            img_ax = res_ax["image"]
            res_cor = self.transforms(image=img_cor)
            img_cor = res_cor["image"]
        else:
            # Fallback to simple tensor conversion
            t = ToTensorV2()
            img_ax = t(image=img_ax)["image"].float() / 255.0
            img_cor = t(image=img_cor)["image"].float() / 255.0

        # 3. Extract Tabular Features & Target
        if self.mode == "test":
            # Test metadata has baseline columns merged
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
            percent = row["Baseline_Percent"]
            baseline_fvc = row["Baseline_FVC"]

            # Relative week for prediction
            # Predict_Week is the target week, Baseline_Week is when CT was taken
            rel_week = row["Predict_Week"] - row["Baseline_Week"]

            target_fvc = 0.0  # Dummy
            patient_week_id = row["Patient_Week"]

        else:
            # Train/Val metadata
            age = row["Age"]
            sex = row["Sex"]
            smoke = row["SmokingStatus"]
            percent = row["Percent"]

            # Lookup baseline info
            baseline_fvc = self.baseline_lookup.get(patient_id, 2000)
            baseline_week = self.baseline_week_lookup.get(patient_id, 0)

            # Relative week
            rel_week = row["Weeks"] - baseline_week

            target_fvc = row["FVC"]
            patient_week_id = f"{patient_id}_{row['Weeks']}"

        # 4. Encode Tabular Data
        # Scaling (approximate standardization based on dataset stats)
        age_sc = (age - 65.0) / 15.0
        pct_sc = (percent - 80.0) / 20.0

        # Categorical Encoding
        sex_enc = self.sex_map.get(sex, 0)
        smoke_enc = self.smoke_map.get(smoke, 0)

        # One-Hot Encoding for Smoking (3 classes)
        smoke_oh = [0.0, 0.0, 0.0]
        smoke_oh[smoke_enc] = 1.0

        # Sex Feature (Binary)
        sex_feat = 1.0 if sex_enc == 1 else 0.0

        # Construct Feature Vector: [Age, Percent, Sex, Smoke_Ex, Smoke_Nev, Smoke_Cur]
        # Dimension: 1 + 1 + 1 + 3 = 6
        tab_vec = np.array([age_sc, pct_sc, sex_feat] + smoke_oh, dtype=np.float32)

        # Baseline FVC is passed separately for the skip connection
        # We also scale it for numerical stability in the network
        base_fvc_sc = (baseline_fvc - 2500.0) / 1000.0

        return {
            "img_ax": img_ax,  # (3, 224, 224)
            "img_cor": img_cor,  # (3, 224, 224)
            "tab_vec": torch.from_numpy(tab_vec),  # (6,)
            "rel_week": torch.tensor(rel_week, dtype=torch.float32),
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "baseline_fvc_sc": torch.tensor(base_fvc_sc, dtype=torch.float32),
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "patient_week": patient_week_id,
        }
