import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


class DICOMProcessor:
    """
    Handles loading, processing, and caching of DICOM images.
    Generates Fixed Overlapping Orthogonal Tri-Slabs.
    """

    def __init__(self, cache_dir, img_size=224, load_cached=True):
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.load_cached = load_cached

        # Lung window settings (standard for lung CT)
        self.hu_min = -1000
        self.hu_max = 400

    def get_images(self, patient_id, dicom_dir):
        """
        Returns (axial_img, coronal_img).
        Shape: (3, H, W) normalized to [0, 1].
        """
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try loading from cache
        if (
            self.load_cached
            and os.path.exists(axial_path)
            and os.path.exists(coronal_path)
        ):
            try:
                axial = np.load(axial_path).astype(np.float32)
                coronal = np.load(coronal_path).astype(np.float32)
                return axial, coronal
            except Exception:
                pass  # Fallback to processing if file is corrupt

        # 2. Process from scratch
        # Load volume
        volume = self._load_scan(dicom_dir)  # (Z, Y, X)

        # Generate Tri-Slabs
        axial = self._generate_tri_slab(volume, axis=0)  # Split along Z (Axial)
        coronal = self._generate_tri_slab(volume, axis=1)  # Split along Y (Coronal)

        # Save to cache
        np.save(axial_path, axial)
        np.save(coronal_path, coronal)

        return axial, coronal

    def _load_scan(self, path):
        full_path = os.path.join(Config.input_root, path)
        if not os.path.exists(full_path):
            # Fallback for missing directories
            return np.zeros((10, 512, 512), dtype=np.float32)

        slices = []
        # List all dcm files
        try:
            files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
        except FileNotFoundError:
            return np.zeros((10, 512, 512), dtype=np.float32)

        if not files:
            return np.zeros((10, 512, 512), dtype=np.float32)

        for s in files:
            try:
                ds = pydicom.dcmread(os.path.join(full_path, s))
                slices.append(ds)
            except:
                continue

        if not slices:
            return np.zeros((10, 512, 512), dtype=np.float32)

        # Sort by ImagePositionPatient Z
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            # Fallback if no position info, sort by filename number
            slices.sort(
                key=lambda x: (
                    int(os.path.splitext(os.path.basename(x.filename))[0])
                    if os.path.splitext(os.path.basename(x.filename))[0].isdigit()
                    else x.filename
                )
            )

        # Create volume
        # Safely load pixel arrays, handling decompression errors (Cite debug_lesson_2)
        valid_arrays = []
        for s in slices:
            try:
                arr = s.pixel_array.astype(np.float32)
                valid_arrays.append(arr)
            except (RuntimeError, Exception):
                # Skip slices that cannot be decompressed (missing codecs) or are corrupt
                continue

        if not valid_arrays:
            # Return placeholder if no slices could be loaded (Cite debug_lesson_8)
            return np.zeros((10, 512, 512), dtype=np.float32)

        # Handle varying slice sizes by resizing to the median shape if necessary
        try:
            image = np.stack(valid_arrays)
        except ValueError:
            target_shape = valid_arrays[len(valid_arrays) // 2].shape
            resized_slices = []
            for arr in valid_arrays:
                if arr.shape != target_shape:
                    arr = cv2.resize(arr, (target_shape[1], target_shape[0]))
                resized_slices.append(arr)
            image = np.stack(resized_slices)

        # Convert to HU
        slope = getattr(slices[0], "RescaleSlope", 1)
        intercept = getattr(slices[0], "RescaleIntercept", -1024)

        image = image * slope + intercept

        # Clip to lung window and normalize
        image = np.clip(image, self.hu_min, self.hu_max)
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)

        return image.astype(np.float32)

    def _generate_tri_slab(self, volume, axis):
        """
        Generates 3-channel image using overlapping maximum intensity projections.
        axis=0: Axial (split Z, project Z) -> Result (3, Y, X) -> Resize -> (3, 224, 224)
        axis=1: Coronal (split Y, project Y) -> Result (3, Z, X) -> Resize -> (3, 224, 224)
        """
        # Move split axis to front (0)
        if axis != 0:
            volume = np.swapaxes(volume, 0, axis)

        depth = volume.shape[0]

        # Define slabs
        # Overlap 15% of a third
        slab_depth = depth / 3.0
        overlap = int(slab_depth * 0.15)

        # Slab 1: 0 to 33% + overlap
        idx1_start = 0
        idx1_end = int(slab_depth + overlap)

        # Slab 2: 33% - overlap to 66% + overlap
        idx2_start = int(slab_depth - overlap)
        idx2_end = int(2 * slab_depth + overlap)

        # Slab 3: 66% - overlap to 100%
        idx3_start = int(2 * slab_depth - overlap)
        idx3_end = depth

        # Ensure indices are within bounds
        idx1_end = min(idx1_end, depth)
        idx2_start = max(0, idx2_start)
        idx2_end = min(idx2_end, depth)
        idx3_start = max(0, idx3_start)

        # Compute MIPs
        def get_mip(start, end):
            if start >= end:
                return np.zeros(volume.shape[1:], dtype=np.float32)
            slab = volume[start:end, :, :]
            return np.max(slab, axis=0)

        c1 = get_mip(idx1_start, idx1_end)
        c2 = get_mip(idx2_start, idx2_end)
        c3 = get_mip(idx3_start, idx3_end)

        # Stack channels
        img = np.stack([c1, c2, c3], axis=-1)  # (H, W, 3)

        # Resize to target size
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )

        # Transpose to (3, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))

        return img


class LungDataset(Dataset):
    def __init__(self, mode="train", transform=None, debug=False):
        self.mode = mode
        self.debug = debug

        # Load metadata
        if mode == "train":
            self.df = pd.read_csv(Config.train_file)
        elif mode == "val":
            self.df = pd.read_csv(Config.val_file)
        elif mode == "test":
            self.df = pd.read_csv(Config.test_file)

        if self.debug:
            self.df = self.df.head(Config.debug_sample_size)

        # Initialize processor
        self.processor = DICOMProcessor(
            cache_dir=Config.cache_dir,
            img_size=Config.img_size,
            load_cached=Config.load_cached_data,
        )

        # Augmentations
        if transform is None:
            if mode == "train":
                # Spatial only: Flips, Shifts, Rotations. No intensity changes.
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            scale_limit=0,
                            rotate_limit=10,
                            shift_limit=0.1,
                            p=0.5,
                            border_mode=cv2.BORDER_CONSTANT,
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([ToTensorV2()])
        else:
            self.transform = transform

        # Precompute tabular normalization stats
        self.age_min, self.age_max = 50.0, 90.0
        self.pct_min, self.pct_max = 30.0, 150.0

        # Build baseline lookup for training/val modes to anchor the trajectory
        self.baseline_lookup = {}
        if mode in ["train", "val"]:
            # Group by patient and find FVC at min(abs(Weeks))
            for pid, group in self.df.groupby("Patient"):
                # Find row closest to week 0
                closest_idx = group["Weeks"].abs().idxmin()
                baseline_fvc = group.loc[closest_idx, "FVC"]
                self.baseline_lookup[pid] = baseline_fvc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Get Images
        dicom_dir = row["dicom_dir"]
        axial, coronal = self.processor.get_images(patient_id, dicom_dir)

        # Convert to (H, W, C) for Albumentations
        axial_np = np.transpose(axial, (1, 2, 0))
        coronal_np = np.transpose(coronal, (1, 2, 0))

        # Apply augmentations
        # Apply independently to views for regularization
        aug_axial = self.transform(image=axial_np)["image"]
        aug_coronal = self.transform(image=coronal_np)["image"]

        # 2. Tabular Features
        # Target Dim: 7 -> Age(1), Sex(2), Smoking(3), Percent(1)

        # Extract features (handle column name differences between train/test)
        if self.mode == "test":
            age = float(row["Baseline_Age"])
            pct = float(row["Baseline_Percent"])
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
        else:
            age = float(row["Age"])
            pct = float(row["Percent"])
            sex = row["Sex"]
            smoke = row["SmokingStatus"]

        # Normalize/Encode
        age_norm = (age - self.age_min) / (self.age_max - self.age_min)
        pct_norm = (pct - self.pct_min) / (self.pct_max - self.pct_min)

        # Sex One-Hot
        sex_vec = [1.0, 0.0] if sex == "Male" else [0.0, 1.0]

        # Smoking One-Hot
        if smoke == "Ex-smoker":
            smoke_vec = [1.0, 0.0, 0.0]
        elif smoke == "Never smoked":
            smoke_vec = [0.0, 1.0, 0.0]
        else:  # Currently smokes
            smoke_vec = [0.0, 0.0, 1.0]

        tabular = np.array(
            [age_norm] + sex_vec + smoke_vec + [pct_norm], dtype=np.float32
        )

        # 3. Targets and Week
        if self.mode == "test":
            week = float(row["Predict_Week"])
            baseline_week = float(row["Baseline_Week"])
            baseline_fvc = float(row["Baseline_FVC"])

            rel_week = week - baseline_week

            return {
                "axial": aug_axial,
                "coronal": aug_coronal,
                "tabular": torch.tensor(tabular, dtype=torch.float32),
                "rel_week": torch.tensor(rel_week, dtype=torch.float32),
                "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
                "patient_week_id": row["Patient_Week"],
            }

        else:
            # Train/Val
            fvc = float(row["FVC"])
            rel_week = float(row["Weeks"])
            baseline_fvc = self.baseline_lookup.get(
                patient_id, 2500.0
            )  # Fallback if lookup fails

            return {
                "axial": aug_axial,
                "coronal": aug_coronal,
                "tabular": torch.tensor(tabular, dtype=torch.float32),
                "rel_week": torch.tensor(rel_week, dtype=torch.float32),
                "fvc": torch.tensor(fvc, dtype=torch.float32),
                "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            }
