import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Try importing pydicom, handle if missing (though required for this task)
try:
    import pydicom
except ImportError:
    pydicom = None
    print("Warning: pydicom not installed. DICOM processing will fail.")


class DataProcessor:
    """
    Handles the loading, processing, and caching of DICOM images into Tri-Slab inputs.
    """

    @staticmethod
    def get_lung_window(img, min_bound=-1000, max_bound=400):
        """
        Applies lung windowing to Hounsfield Units.
        """
        img = (img - min_bound) / (max_bound - min_bound)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    @staticmethod
    def generate_tri_slab(volume, axis=0, overlap=0.15):
        """
        Splits a 3D volume into 3 overlapping slabs along the specified axis
        and computes the Maximum Intensity Projection (MIP) for each.

        Args:
            volume: 3D numpy array.
            axis: Axis to split along (0 for Depth/Z, 1 for Height/Y).
            overlap: Fraction of overlap between slabs.

        Returns:
            A 3-channel 2D image (H, W, 3) where channels are MIPs of the slabs.
        """
        # Move the target splitting axis to 0 for uniform processing
        if axis != 0:
            volume = np.moveaxis(volume, axis, 0)

        depth = volume.shape[0]
        if depth < 3:
            # Fallback for extremely thin volumes: repeat the volume
            mip = np.max(volume, axis=0)
            return np.stack([mip, mip, mip], axis=-1)

        # Calculate slab boundaries
        # We want 3 slabs covering the range [0, depth]
        # Slab 1: 0 -> 1/3 + overlap
        # Slab 2: 1/3 -> 2/3 + overlap
        # Slab 3: 2/3 -> 1

        third = depth / 3.0
        overlap_px = int(depth * overlap)

        start1, end1 = 0, int(third) + overlap_px
        start2, end2 = int(third) - overlap_px, int(2 * third) + overlap_px
        start3, end3 = int(2 * third) - overlap_px, depth

        # Clamp indices
        start2 = max(0, start2)
        start3 = max(0, start3)
        end1 = min(depth, end1)
        end2 = min(depth, end2)

        # Extract slabs
        slab1 = volume[start1:end1, :, :]
        slab2 = volume[start2:end2, :, :]
        slab3 = volume[start3:end3, :, :]

        # Compute MIPs
        mip1 = np.max(slab1, axis=0) if slab1.size > 0 else np.zeros_like(volume[0])
        mip2 = np.max(slab2, axis=0) if slab2.size > 0 else np.zeros_like(volume[0])
        mip3 = np.max(slab3, axis=0) if slab3.size > 0 else np.zeros_like(volume[0])

        # Stack to create (H, W, 3)
        img = np.stack([mip1, mip2, mip3], axis=-1)
        return img

    @staticmethod
    def load_scan(path):
        """
        Loads all DICOM files from a directory, sorts them by Z-position,
        and returns a 3D volume resized to (D, 224, 224).
        """
        if pydicom is None:
            raise ImportError("pydicom is required to load scans.")

        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            # Return dummy volume if no files found (should not happen based on EDA)
            return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                slices.append(dcm)
            except:
                continue

        # Sort slices
        # Try sorting by ImagePositionPatient[2], fallback to InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            try:
                slices.sort(key=lambda x: float(x.InstanceNumber))
            except AttributeError:
                pass  # Keep original order if sorting fails

        # Extract images and resize
        img_list = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)

            # Convert to HU
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = img * slope + intercept

            # Resize to native resolution (224x224)
            # Note: cv2.resize expects (W, H)
            if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
                img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

            img_list.append(img)

        volume = np.stack(img_list, axis=0)  # (D, H, W)
        return volume

    @classmethod
    def process_patient(cls, patient_id, dicom_dir, cache_dir, load_cached=True):
        """
        Orchestrates the processing of a patient's CT scan.
        Returns Axial and Coronal Tri-Slab images.
        """
        axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try Loading from Cache
        if load_cached and os.path.exists(axial_path) and os.path.exists(coronal_path):
            try:
                ax = np.load(axial_path)
                cor = np.load(coronal_path)
                return ax, cor
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process from Scratch
        full_path = os.path.join(Config.INPUT_ROOT, dicom_dir)
        try:
            volume = cls.load_scan(full_path)

            # Apply Lung Window
            volume = cls.get_lung_window(volume)

            # Generate Axial Tri-Slab (Split along Depth/Z - axis 0)
            img_axial = cls.generate_tri_slab(
                volume, axis=0, overlap=Config.SLAB_OVERLAP
            )

            # Generate Coronal Tri-Slab (Split along Height/Y - axis 1)
            img_coronal = cls.generate_tri_slab(
                volume, axis=1, overlap=Config.SLAB_OVERLAP
            )

            # Resize Coronal view to fixed resolution (Depth -> 224)
            if (
                img_coronal.shape[0] != Config.IMG_SIZE
                or img_coronal.shape[1] != Config.IMG_SIZE
            ):
                img_coronal = cv2.resize(
                    img_coronal, (Config.IMG_SIZE, Config.IMG_SIZE)
                )

            # 3. Save to Cache
            np.save(axial_path, img_axial)
            np.save(coronal_path, img_coronal)

            return img_axial, img_coronal

        except Exception as e:
            # Return zeros in case of catastrophic failure to avoid crashing the pipeline
            # print(f"Error processing {patient_id}: {e}")
            dummy = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
            return dummy, dummy


class OSICDataset(Dataset):
    def __init__(self, csv_path, mode="train", transform=None, load_cached=True):
        self.mode = mode
        self.transform = transform
        self.load_cached = load_cached

        # Load Metadata
        self.df = pd.read_csv(csv_path)

        # Cite debug_lesson_9: Anticipate Schema Differences Between Training and Inference Data
        if self.mode == "test":
            self.df = self.df.rename(
                columns={
                    "Baseline_Age": "Age",
                    "Baseline_Sex": "Sex",
                    "Baseline_SmokingStatus": "SmokingStatus",
                }
            )

        # Pre-process Tabular Data
        self._prepare_tabular_data()

    def _prepare_tabular_data(self):
        """
        Prepares tabular features and identifies baseline values.
        """
        # Normalize Age (approx range 50-90)
        self.df["Age_norm"] = (self.df["Age"] - 50) / 50.0

        # One-Hot Encode Sex
        self.df["Sex_Male"] = (self.df["Sex"] == "Male").astype(float)
        self.df["Sex_Female"] = (self.df["Sex"] == "Female").astype(float)

        # One-Hot Encode SmokingStatus
        self.df["Smoke_Ex"] = (self.df["SmokingStatus"] == "Ex-smoker").astype(float)
        self.df["Smoke_Never"] = (self.df["SmokingStatus"] == "Never smoked").astype(
            float
        )
        self.df["Smoke_Cur"] = (self.df["SmokingStatus"] == "Currently smokes").astype(
            float
        )

        # Handle Baseline Logic
        if self.mode in ["train", "val"]:
            # For training, we need to identify the baseline for each patient
            # Baseline is defined as the visit closest to Week 0 (min abs(Weeks))
            # We create a lookup for baseline features

            # Calculate absolute weeks to find baseline
            self.df["Abs_Weeks"] = self.df["Weeks"].abs()

            # Group by patient and find the row with min Abs_Weeks
            # We use sort_values + drop_duplicates to get the baseline row per patient
            baseline_df = self.df.sort_values("Abs_Weeks").drop_duplicates(
                "Patient", keep="first"
            )

            # Select relevant baseline columns
            baseline_cols = ["Patient", "FVC", "Percent", "Weeks"]
            baseline_df = baseline_df[baseline_cols].rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Weeks": "Baseline_Week",
                }
            )

            # Merge baseline info back to the main dataframe
            self.df = self.df.merge(baseline_df, on="Patient", how="left")

        elif self.mode == "test":
            # Test set (from metadata/test.csv) already has Baseline_ columns
            # Ensure naming consistency
            pass

        # Normalize Percent (approx range 30-150)
        self.df["Baseline_Percent_norm"] = (self.df["Baseline_Percent"] - 50) / 100.0

        # Normalize Baseline FVC (approx range 1000-6000)
        self.df["Baseline_FVC_norm"] = (self.df["Baseline_FVC"] - 2000) / 2000.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial and Coronal)
        dicom_dir = row["dicom_dir"]
        img_axial, img_coronal = DataProcessor.process_patient(
            patient_id, dicom_dir, Config.CACHE_DIR, self.load_cached
        )

        # 2. Apply Augmentations (Spatial Only)
        if self.transform:
            # Apply same transform to both views?
            # Usually independent is fine, but here they represent the same patient.
            # However, they are different views. Independent aug is acceptable for robustness.
            res_ax = self.transform(image=img_axial)
            img_axial = res_ax["image"]

            res_cor = self.transform(image=img_coronal)
            img_coronal = res_cor["image"]

        # Convert to Tensor (C, H, W)
        img_axial = torch.tensor(img_axial, dtype=torch.float32).permute(2, 0, 1)
        img_coronal = torch.tensor(img_coronal, dtype=torch.float32).permute(2, 0, 1)

        # 3. Prepare Tabular Meta Vector
        # Features: [Age, Sex_M, Sex_F, Smoke_Ex, Smoke_Nev, Smoke_Cur, Baseline_Percent, Baseline_FVC]
        meta_features = [
            row["Age_norm"],
            row["Sex_Male"],
            row["Sex_Female"],
            row["Smoke_Ex"],
            row["Smoke_Never"],
            row["Smoke_Cur"],
            row["Baseline_Percent_norm"],
            row["Baseline_FVC_norm"],
        ]
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32)

        # 4. Prepare Trajectory Info
        # Week Diff: Current Week - Baseline Week
        if self.mode == "test":
            current_week = row["Predict_Week"]
        else:
            current_week = row["Weeks"]

        week_diff = current_week - row["Baseline_Week"]

        # 5. Prepare Target
        if self.mode == "test":
            target = 0.0  # Dummy
        else:
            target = row["FVC"]

        return {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "meta": meta_tensor,
            "week_diff": torch.tensor(week_diff, dtype=torch.float32),
            "baseline_fvc": torch.tensor(row["Baseline_FVC"], dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_id": patient_id,
            "week": torch.tensor(current_week, dtype=torch.float32),
        }


def get_img_transform(mode="train"):
    """
    Returns Albumentations transforms.
    Strictly spatial augmentations for training, no intensity changes.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
            ]
        )
    else:
        return None


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates and returns Train, Val, and Test dataloaders.
    """
    set_seed(Config.SEED)

    # Train Loader
    train_ds = OSICDataset(
        Config.META_TRAIN,
        mode="train",
        transform=get_img_transform("train"),
        load_cached=Config.LOAD_CACHED_DATA,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Val Loader
    val_ds = OSICDataset(
        Config.META_VAL, mode="val", transform=None, load_cached=Config.LOAD_CACHED_DATA
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Loader
    test_ds = OSICDataset(
        Config.META_TEST,
        mode="test",
        transform=None,
        load_cached=Config.LOAD_CACHED_DATA,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
