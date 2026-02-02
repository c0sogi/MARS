import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Attempt to import pydicom; handle gracefully if missing (though required for this task)
try:
    import pydicom
except ImportError:
    pydicom = None
    print("Warning: pydicom not found. DICOM processing will fail.")

from library.config import Config
from library.utils import seed_everything


class DICOMProcessor:
    """
    Handles loading DICOM files, converting to HU, windowing, and generating
    Fixed Overlapping Orthogonal Tri-Slabs (Axial and Coronal).
    """

    def __init__(self, img_size=224, num_slabs=3, overlap=0.15):
        self.img_size = img_size
        self.num_slabs = num_slabs
        self.overlap = overlap

    def get_lung_window(self, img):
        """Applies standard lung windowing (L=-600, W=1500) and normalizes to 0-255."""
        window_center = -600
        window_width = 1500
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        # Normalize to 0-255
        img = (img - img_min) / (img_max - img_min) * 255.0
        return img.astype(np.uint8)

    def load_scan(self, path):
        """Loads all DICOM files from a directory and stacks them into a 3D volume."""
        if not pydicom:
            raise ImportError("pydicom is required to read DICOM files.")

        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            return None

        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except Exception:
                continue

        if not slices:
            return None

        # Sort by ImagePositionPatient Z (index 2) or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Extract pixel data and convert to HU
        image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

        # Apply Intercept and Slope if available
        if hasattr(slices[0], "RescaleIntercept") and hasattr(
            slices[0], "RescaleSlope"
        ):
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope
            image = image * slope + intercept

        return image

    def create_tri_slab(self, volume, axis=0):
        """
        Creates a 3-channel image by splitting the volume into 3 overlapping slabs
        along the specified axis and computing the MIP for each.

        Args:
            volume: 3D numpy array (D, H, W)
            axis: 0 for Axial (split D), 1 for Coronal (split H)
        """
        # If Coronal, we permute so the split axis is 0
        if axis == 1:
            # (D, H, W) -> (H, D, W)
            volume = np.transpose(volume, (1, 0, 2))

        depth = volume.shape[0]
        slab_depth = depth / self.num_slabs
        overlap_px = slab_depth * self.overlap

        channels = []
        for i in range(self.num_slabs):
            # Define boundaries with overlap
            start = max(0, int(i * slab_depth - overlap_px))
            end = min(depth, int((i + 1) * slab_depth + overlap_px))

            # Extract slab
            slab = volume[start:end, :, :]

            # Compute MIP (Maximum Intensity Projection)
            if slab.shape[0] > 0:
                mip = np.max(slab, axis=0)
            else:
                mip = np.zeros((volume.shape[1], volume.shape[2]), dtype=volume.dtype)

            channels.append(mip)

        # Stack to create (H, W, 3)
        img = np.stack(channels, axis=-1)
        return img

    def process_patient(self, dicom_dir):
        """
        Full pipeline: Load -> Window -> Tri-Slab (Axial & Coronal) -> Resize.
        Returns: (axial_img, coronal_img) both (224, 224, 3)
        """
        volume = self.load_scan(dicom_dir)
        if volume is None:
            # Return black images if loading fails
            return np.zeros(
                (self.img_size, self.img_size, 3), dtype=np.uint8
            ), np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Apply lung windowing to the whole volume first for efficiency
        volume = self.get_lung_window(volume)

        # Generate Axial Tri-Slab (Split along Z/Depth - axis 0)
        axial = self.create_tri_slab(volume, axis=0)

        # Generate Coronal Tri-Slab (Split along Y/Height - axis 1)
        coronal = self.create_tri_slab(volume, axis=1)

        # Resize
        axial = cv2.resize(
            axial, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )
        coronal = cv2.resize(
            coronal, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        return axial, coronal


def process_and_cache_images(
    patient_ids, metadata_df, input_dir, cache_dir, load_cached=True
):
    """
    Iterates through patients, processes their CT scans, and caches the results.
    """
    processor = DICOMProcessor(img_size=Config.IMG_SIZE)
    os.makedirs(cache_dir, exist_ok=True)

    # Filter unique patients
    unique_patients = np.unique(patient_ids)

    for pid in unique_patients:
        ax_path = os.path.join(cache_dir, f"{pid}_axial.npy")
        cor_path = os.path.join(cache_dir, f"{pid}_coronal.npy")

        # Check cache
        if load_cached and os.path.exists(ax_path) and os.path.exists(cor_path):
            continue

        # Get DICOM directory
        # We find the relative path from the metadata
        # Assuming metadata_df has 'Patient' and 'dicom_dir'
        subset = metadata_df[metadata_df["Patient"] == pid]
        if len(subset) == 0:
            continue

        rel_path = subset.iloc[0]["dicom_dir"]
        full_path = os.path.join(input_dir, rel_path)

        # Process
        try:
            axial, coronal = processor.process_patient(full_path)
            np.save(ax_path, axial)
            np.save(cor_path, coronal)
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            # Save zeros to prevent crash in Dataset
            zeros = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            np.save(ax_path, zeros)
            np.save(cor_path, zeros)


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

        # Pre-calculate normalization stats for tabular data
        # Note: In a real scenario, these should be fixed or fitted on train only.
        # Here we use approximate fixed scalars for stability.
        self.age_mean = 65.0
        self.age_std = 15.0
        self.pct_mean = 80.0
        self.pct_std = 20.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Load Images
        ax_path = os.path.join(self.cache_dir, f"{pid}_axial.npy")
        cor_path = os.path.join(self.cache_dir, f"{pid}_coronal.npy")

        try:
            img_ax = np.load(ax_path)
            img_cor = np.load(cor_path)
        except FileNotFoundError:
            # Fallback
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # 2. Augmentations (Spatial Only)
        if self.transform:
            # Apply same transform to both or independent?
            # Usually independent is fine for regularization, but geometric consistency might differ.
            # Given they are different views, independent is acceptable.
            img_ax = self.transform(image=img_ax)["image"]
            img_cor = self.transform(image=img_cor)["image"]
        else:
            # Just normalize and to tensor
            t = A.Compose(
                [
                    A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                    ToTensorV2(),
                ]
            )
            img_ax = t(image=img_ax)["image"]
            img_cor = t(image=img_cor)["image"]

        # 3. Tabular Features (Prior)
        # We need Baseline Age, Sex, Smoking, Percent
        # In train/val df, 'Age'/'Percent' are current. We should use Baseline if possible.
        # However, for simplicity and strong signal, we use the values provided in the row
        # (which are very close to baseline usually) or specifically Baseline columns if available.

        if "Baseline_Age" in row:
            age = row["Baseline_Age"]
            pct = row["Baseline_Percent"]
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
        else:
            age = row["Age"]
            pct = row["Percent"]
            sex = row["Sex"]
            smoke = row["SmokingStatus"]

        # Normalize Numerical
        age_norm = (age - self.age_mean) / self.age_std
        pct_norm = (pct - self.pct_mean) / self.pct_std

        # Encode Categorical
        # Sex: Male=0, Female=1
        sex_enc = 0.0 if sex == "Male" else 1.0

        # Smoking: Ex-smoker=0, Never smoked=0.5, Currently smokes=1.0 (Ordinal-ish mapping)
        if smoke == "Ex-smoker":
            smoke_enc = 0.0
        elif smoke == "Never smoked":
            smoke_enc = 0.5
        else:
            smoke_enc = 1.0

        tabular = torch.tensor(
            [age_norm, pct_norm, sex_enc, smoke_enc], dtype=torch.float32
        )

        # 4. Metadata & Target
        if self.mode in ["train", "val"]:
            # Target is FVC
            target = torch.tensor([row["FVC"]], dtype=torch.float32)

            # Meta: Baseline_FVC, Delta_Week
            # We need to find the baseline FVC for this patient.
            # In get_dataloaders, we will ensure 'Baseline_FVC' and 'Baseline_Week' are added to the DF.
            baseline_fvc = row["Baseline_FVC"]
            delta_week = row["Weeks"] - row["Baseline_Week"]

            meta = torch.tensor([baseline_fvc, delta_week], dtype=torch.float32)

            return {
                "image_axial": img_ax,
                "image_coronal": img_cor,
                "tabular": tabular,
                "meta": meta,
                "target": target,
            }

        else:  # Test mode
            # Target is dummy
            target = torch.tensor([0.0], dtype=torch.float32)

            # Meta: Baseline_FVC, Delta_Week
            baseline_fvc = row["Baseline_FVC"]
            delta_week = row["Predict_Week"] - row["Baseline_Week"]

            meta = torch.tensor([baseline_fvc, delta_week], dtype=torch.float32)

            # We also return Patient_Week for submission mapping
            patient_week = row["Patient_Week"]

            return {
                "image_axial": img_ax,
                "image_coronal": img_cor,
                "tabular": tabular,
                "meta": meta,
                "target": target,
                "patient_week": patient_week,
            }


def get_dataloaders(batch_size=None, load_cached_data=True, debug=False):
    """
    Main entry point to prepare data.
    1. Loads metadata.
    2. Identifies baselines for train/val.
    3. Caches images.
    4. Returns DataLoaders.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        # Keep test intact or subset? Usually keep test small in debug if needed,
        # but here we might want to check full pipeline.

    # 2. Prepare Baseline Info for Train/Val
    # We need to add 'Baseline_FVC' and 'Baseline_Week' to every row.
    # Strategy: For each patient, pick the row with min(abs(Weeks)).

    def add_baseline_info(df):
        # Identify baseline rows
        # Sort by patient and absolute weeks to find the one closest to 0
        df["Abs_Weeks"] = df["Weeks"].abs()
        df = df.sort_values(["Patient", "Abs_Weeks"])

        # Group by Patient and take first
        baselines = df.groupby("Patient").first().reset_index()

        # Select relevant cols
        baselines = baselines[
            ["Patient", "FVC", "Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
        ]
        baselines = baselines.rename(
            columns={
                "FVC": "Baseline_FVC",
                "Weeks": "Baseline_Week",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
            }
        )

        # Merge back
        # Drop original df cols that might conflict if we want strict baseline prior?
        # The Dataset logic checks for 'Baseline_' cols first.
        df = df.drop(columns=["Abs_Weeks"])
        df = pd.merge(df, baselines, on="Patient", how="left")
        return df

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # Test DF already has Baseline info from metadata generation script

    # 3. Cache Images
    # Collect all patients
    all_patients = np.concatenate(
        [
            train_df["Patient"].unique(),
            val_df["Patient"].unique(),
            test_df["Patient"].unique(),
        ]
    )

    # Combine dfs for path lookup
    combined_meta = pd.concat(
        [
            train_df[["Patient", "dicom_dir"]],
            val_df[["Patient", "dicom_dir"]],
            test_df[["Patient", "dicom_dir"]],
        ]
    ).drop_duplicates()

    print("Processing and caching images...")
    process_and_cache_images(
        all_patients,
        combined_meta,
        Config.INPUT_DIR,
        Config.CACHE_DIR,
        load_cached=load_cached_data,
    )

    # 4. Augmentations
    # Spatial only: Flips, ShiftScaleRotate. No brightness/contrast.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
            ToTensorV2(),
        ]
    )

    # 5. Create Datasets
    train_dataset = OSICDataset(
        train_df, Config.CACHE_DIR, mode="train", transform=train_transform
    )
    val_dataset = OSICDataset(val_df, Config.CACHE_DIR, mode="val", transform=None)
    test_dataset = OSICDataset(test_df, Config.CACHE_DIR, mode="test", transform=None)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
