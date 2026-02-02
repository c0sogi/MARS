import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from library.config import Config


class TriSlabProcessor:
    """
    Handles loading DICOM directories, processing them into 3D volumes,
    and generating Fixed Overlapping Tri-Slabs for Axial and Coronal views.
    """

    def __init__(
        self,
        cache_dir=Config.CACHE_DIR,
        img_size=Config.IMG_SIZE,
        n_slabs=Config.N_SLABS,
        overlap=Config.OVERLAP,
    ):
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.n_slabs = n_slabs
        self.overlap = overlap
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_slab_ranges(self, total_depth):
        """Calculates start and end indices for overlapping slabs."""
        # We want n_slabs covering [0, total_depth] with overlap
        limits = np.linspace(0, total_depth, self.n_slabs + 1)

        ranges = []
        for i in range(self.n_slabs):
            start = limits[i]
            end = limits[i + 1]

            # Apply overlap
            span = end - start
            margin = span * self.overlap

            s_idx = int(max(0, start - margin))
            e_idx = int(min(total_depth, end + margin))

            # Ensure at least 1 slice
            if e_idx <= s_idx:
                e_idx = s_idx + 1

            ranges.append((s_idx, e_idx))

        return ranges

    def process_volume(self, volume, view_plane):
        """
        Processes a 3D volume (D, H, W) into a Tri-Slab image (3, Size, Size).
        view_plane: 'axial' (MIP along D) or 'coronal' (MIP along H after permute).
        """
        # Volume is assumed to be (Z, Y, X) -> (Depth, Height, Width)

        if view_plane == "axial":
            # Axial: Slices are along Z. We resize X, Y.
            D, H, W = volume.shape
            resized_vol = []
            for i in range(D):
                slc = volume[i]
                slc = cv2.resize(slc, (self.img_size, self.img_size))
                resized_vol.append(slc)
            resized_vol = np.array(resized_vol)  # (D, 224, 224)

            ranges = self.get_slab_ranges(D)
            channels = []
            for s, e in ranges:
                slab = resized_vol[s:e, :, :]
                if slab.shape[0] == 0:
                    mip = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                else:
                    mip = np.max(slab, axis=0)
                channels.append(mip)

        elif view_plane == "coronal":
            # Coronal: Plane is X-Z. Y is the depth dimension.
            # Permute to (Y, Z, X) -> (Depth, Height, Width)
            vol_perm = np.transpose(volume, (1, 0, 2))
            D, H, W = vol_perm.shape

            resized_vol = []
            for i in range(D):
                slc = vol_perm[i]
                slc = cv2.resize(slc, (self.img_size, self.img_size))
                resized_vol.append(slc)
            resized_vol = np.array(resized_vol)

            ranges = self.get_slab_ranges(D)
            channels = []
            for s, e in ranges:
                slab = resized_vol[s:e, :, :]
                if slab.shape[0] == 0:
                    mip = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                else:
                    mip = np.max(slab, axis=0)
                channels.append(mip)

        # Stack channels: (3, 224, 224)
        img = np.stack(channels, axis=0)
        return img

    def load_dicom_volume(self, dicom_dir):
        """Reads DICOM files, sorts, converts to HU, windows, and normalizes."""
        files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
        if not files:
            return np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                # Access pixel array to ensure it's readable
                _ = dcm.pixel_array
                slices.append(dcm)
            except:
                continue

        if not slices:
            return np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        # Sort by ImagePositionPatient Z, or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Create volume
        try:
            img_shape = slices[0].pixel_array.shape
            volume = np.zeros(
                (len(slices), img_shape[0], img_shape[1]), dtype=np.float32
            )

            for i, s in enumerate(slices):
                img2d = s.pixel_array.astype(np.float32)

                # Convert to HU
                intercept = getattr(s, "RescaleIntercept", -1024)
                slope = getattr(s, "RescaleSlope", 1)
                img2d = img2d * slope + intercept

                volume[i] = img2d
        except:
            return np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        # Windowing (Lung Window: [-1000, 400])
        min_hu = -1000.0
        max_hu = 400.0

        volume = np.clip(volume, min_hu, max_hu)

        # Normalize to 0-1
        volume = (volume - min_hu) / (max_hu - min_hu)

        return volume

    def get_images(self, patient_id, dicom_dir, load_cached_data=True):
        """
        Returns (axial_img, coronal_img).
        Checks cache first.
        """
        ax_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        cor_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        if load_cached_data and os.path.exists(ax_path) and os.path.exists(cor_path):
            try:
                ax = np.load(ax_path)
                cor = np.load(cor_path)
                return ax, cor
            except:
                pass

        # Process
        full_path = os.path.join(Config.INPUT_ROOT, dicom_dir)
        volume = self.load_dicom_volume(full_path)

        ax = self.process_volume(volume, "axial")
        cor = self.process_volume(volume, "coronal")

        # Save to cache
        np.save(ax_path, ax)
        np.save(cor_path, cor)

        return ax, cor


class LungDataset(Dataset):
    def __init__(self, df, processor, mode="train"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.mode = mode

        # Pre-calculate Baseline Features for training/val data
        # The model requires static baseline priors (Age, Sex, Smoke, Baseline_Percent)
        # and Baseline FVC for trajectory projection.
        if self.mode in ["train", "val"]:
            self.patient_baselines = {}
            for pid, group in self.df.groupby("Patient"):
                # Find row with min absolute weeks (closest to 0)
                base_row = group.loc[group["Weeks"].abs().idxmin()]
                self.patient_baselines[pid] = {
                    "FVC": base_row["FVC"],
                    "Percent": base_row["Percent"],
                    "Age": base_row["Age"],
                    "Sex": base_row["Sex"],
                    "SmokingStatus": base_row["SmokingStatus"],
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]
        dicom_dir = row["dicom_dir"]

        # 1. Load Images
        img_ax, img_cor = self.processor.get_images(
            pid, dicom_dir, load_cached_data=Config.USE_CACHE
        )

        img_ax = torch.tensor(img_ax, dtype=torch.float32)
        img_cor = torch.tensor(img_cor, dtype=torch.float32)

        # 2. Tabular Features & Meta
        if self.mode == "test":
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
            pct = row["Baseline_Percent"]
            base_fvc = row["Baseline_FVC"]
            # Relative week for projection
            week = row["Predict_Week"] - row["Baseline_Week"]
        else:
            # Use pre-calculated baseline features for consistency
            base_data = self.patient_baselines[pid]
            age = base_data["Age"]
            sex = base_data["Sex"]
            smoke = base_data["SmokingStatus"]
            pct = base_data["Percent"]
            base_fvc = base_data["FVC"]
            week = row["Weeks"]  # Relative to baseline

        # Encoding / Normalization
        # Sex: Male=0, Female=1
        sex_val = 0.0 if sex == "Male" else 1.0

        # Smoking: Never=0, Ex=0.5, Current=1.0
        if smoke == "Never smoked":
            smoke_val = 0.0
        elif smoke == "Ex-smoker":
            smoke_val = 0.5
        else:
            smoke_val = 1.0

        # Age: Centered approx (Age - 65) / 15
        age_norm = (float(age) - 65.0) / 15.0

        # Percent: Scaled 0-1 approx
        pct_norm = float(pct) / 100.0

        tabular = torch.tensor(
            [age_norm, sex_val, smoke_val, pct_norm], dtype=torch.float32
        )

        # Meta: [Relative_Week, Baseline_FVC]
        meta = torch.tensor([float(week), float(base_fvc)], dtype=torch.float32)

        if self.mode != "test":
            target = torch.tensor([float(row["FVC"])], dtype=torch.float32)
            return img_ax, img_cor, tabular, meta, target
        else:
            return img_ax, img_cor, tabular, meta, row["Patient_Week"]


def get_dataloaders(debug=False):
    processor = TriSlabProcessor()

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_pats = train_df["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        val_pats = val_df["Patient"].unique()[:5]
        train_df = train_df[train_df["Patient"].isin(train_pats)]
        val_df = val_df[val_df["Patient"].isin(val_pats)]

    train_ds = LungDataset(train_df, processor, mode="train")
    val_ds = LungDataset(val_df, processor, mode="val")

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

    return train_loader, val_loader


def get_test_dataloader():
    processor = TriSlabProcessor()
    test_df = pd.read_csv(Config.TEST_CSV)

    test_ds = LungDataset(test_df, processor, mode="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
