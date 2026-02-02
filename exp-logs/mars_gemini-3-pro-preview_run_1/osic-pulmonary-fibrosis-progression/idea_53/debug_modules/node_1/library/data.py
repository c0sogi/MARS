import os
import sys
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import Config
try:
    from library.config import Config
except ImportError:
    # Fallback for local testing
    sys.path.append(".")
    from library.config import Config


def get_img_custom_transforms(mode="train"):
    """
    Returns albumentations transforms based on Config.
    Strictly adheres to disabling intensity augmentations.
    """
    if mode == "train":
        transforms_dict = Config.get_transforms()
        return A.Compose(
            [
                A.HorizontalFlip(p=transforms_dict.get("horizontal_flip_prob", 0.5)),
                A.ShiftScaleRotate(
                    shift_limit=transforms_dict.get("shift_limit", 0.05),
                    scale_limit=0.1,
                    rotate_limit=transforms_dict.get("rotate_limit", 10),
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
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


def load_scan_tri_slabs(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Loads DICOM scans, generates Fixed Overlapping Orthogonal Tri-Slabs (Axial and Coronal),
    and caches the result as .npy files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path_axial = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    cache_path_coronal = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path_axial) and os.path.exists(cache_path_coronal):
            try:
                img_axial = np.load(cache_path_axial)
                img_coronal = np.load(cache_path_coronal)
                return img_axial, img_coronal
            except Exception:
                pass  # Fallback to recomputing

    # 2. Process from scratch
    try:
        if not os.path.exists(dicom_dir):
            raise FileNotFoundError(f"Directory not found: {dicom_dir}")

        files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        if not files:
            raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

        # Read slices
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(dicom_dir, f))
                slices.append(ds)
            except:
                continue

        if not slices:
            raise ValueError("No valid DICOM slices read.")

        # Sort slices
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Create 3D Volume (Depth, Height, Width)
        images = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = slope * img + intercept
            images.append(img)

        volume = np.stack(images)

        # Helper to process a specific axis into Tri-Slab RGB
        def process_axis(vol_data, axis_idx):
            # axis_idx: 0 for Axial (Depth), 1 for Coronal (Height)
            if axis_idx == 1:
                # Permute to (H, D, W) so slicing is along dim 0
                vol_view = np.transpose(vol_data, (1, 0, 2))
            else:
                vol_view = vol_data

            num_slices = vol_view.shape[0]
            if num_slices == 0:
                return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

            # Slab boundaries: 0-33%, 33-66%, 66-100% with 15% overlap
            seg_size = num_slices / 3.0
            overlap = seg_size * Config.OVERLAP_RATIO

            # Define slabs
            slabs_indices = [
                (0, int(np.ceil(seg_size + overlap))),
                (
                    int(np.floor(seg_size - overlap)),
                    int(np.ceil(2 * seg_size + overlap)),
                ),
                (int(np.floor(2 * seg_size - overlap)), num_slices),
            ]

            channels = []
            for start, end in slabs_indices:
                # Clamp
                start = max(0, start)
                end = min(num_slices, end)

                if start >= end:
                    slab_mip = np.zeros(
                        (vol_view.shape[1], vol_view.shape[2]), dtype=np.float32
                    )
                else:
                    slab = vol_view[start:end, :, :]
                    slab_mip = np.max(slab, axis=0)

                # Normalize HU
                slab_mip = np.clip(slab_mip, Config.HU_MIN, Config.HU_MAX)
                slab_mip = (slab_mip - Config.HU_MIN) / (Config.HU_MAX - Config.HU_MIN)
                slab_mip = (slab_mip * 255).astype(np.uint8)

                # Resize
                slab_resized = cv2.resize(
                    slab_mip,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                channels.append(slab_resized)

            return np.stack(channels, axis=-1)

        img_axial = process_axis(volume, 0)
        img_coronal = process_axis(volume, 1)

    except Exception:
        # Fallback for errors: return black images
        img_axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        img_coronal = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    # 3. Save to cache
    try:
        np.save(cache_path_axial, img_axial)
        np.save(cache_path_coronal, img_coronal)
    except Exception:
        pass

    return img_axial, img_coronal


class LungDataset(Dataset):
    def __init__(self, df, root_dir, cache_dir, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

        # Normalization constants (approximate from EDA)
        self.age_mean = 67.0
        self.age_std = 15.0
        self.pct_mean = 77.0
        self.pct_std = 20.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Load Images
        dicom_path = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])
        img_axial, img_coronal = load_scan_tri_slabs(
            patient_id, dicom_path, self.cache_dir, load_cached_data=True
        )

        # Apply Transforms
        if self.transform:
            res_ax = self.transform(image=img_axial)
            img_axial_t = res_ax["image"]
            res_cor = self.transform(image=img_coronal)
            img_coronal_t = res_cor["image"]
        else:
            img_axial_t = torch.from_numpy(img_axial.transpose(2, 0, 1)).float() / 255.0
            img_coronal_t = (
                torch.from_numpy(img_coronal.transpose(2, 0, 1)).float() / 255.0
            )

        # Tabular Features
        if self.mode == "test":
            pct = row["Baseline_Percent"]
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
            base_fvc = row["Baseline_FVC"]
            # Relative week for prediction
            relative_week = row["Predict_Week"] - row["Baseline_Week"]
        else:
            pct = row["Percent"]
            age = row["Age"]
            sex = row["Sex"]
            smoke = row["SmokingStatus"]
            # For training, we inject Baseline_FVC in get_dataloaders
            base_fvc = row.get("Baseline_FVC", 2000.0)
            relative_week = row["Weeks"]

        # Normalize Features
        pct_norm = (float(pct) - self.pct_mean) / self.pct_std
        age_norm = (float(age) - self.age_mean) / self.age_std
        sex_val = 0.0 if sex == "Male" else 1.0

        # Smoking: [Ex, Never, Current]
        is_ex = 1.0 if smoke == "Ex-smoker" else 0.0
        is_never = 1.0 if smoke == "Never smoked" else 0.0
        is_cur = 1.0 if smoke == "Currently smokes" else 0.0

        # Tabular Vector: [Percent, Age, Sex, Ex, Never, Current]
        tabular = torch.tensor(
            [pct_norm, age_norm, sex_val, is_ex, is_never, is_cur], dtype=torch.float32
        )

        # Target
        if self.mode != "test":
            target = torch.tensor(row["FVC"], dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)

        return {
            "patient_id": patient_id,
            "image_axial": img_axial_t,
            "image_coronal": img_coronal_t,
            "tabular": tabular,
            "target": target,
            "relative_week": torch.tensor(relative_week, dtype=torch.float32),
            "baseline_fvc": torch.tensor(base_fvc, dtype=torch.float32),
        }


def get_dataloaders(batch_size=None, num_workers=None):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load DataFrames
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Inject Baseline FVC into Train/Val DF for Anchored Trajectory Logic
    def add_baseline_fvc(df):
        patient_baseline = {}
        for pid, group in df.groupby("Patient"):
            # Find row closest to Week 0
            idx_min = group["Weeks"].abs().idxmin()
            base_fvc = group.loc[idx_min, "FVC"]
            patient_baseline[pid] = base_fvc
        df["Baseline_FVC"] = df["Patient"].map(patient_baseline)
        return df

    train_df = add_baseline_fvc(train_df)
    val_df = add_baseline_fvc(val_df)

    # Datasets
    train_ds = LungDataset(
        train_df,
        Config.INPUT_ROOT,
        Config.CACHE_DIR,
        mode="train",
        transform=get_img_custom_transforms("train"),
    )

    val_ds = LungDataset(
        val_df,
        Config.INPUT_ROOT,
        Config.CACHE_DIR,
        mode="val",
        transform=get_img_custom_transforms("val"),
    )

    test_ds = LungDataset(
        test_df,
        Config.INPUT_ROOT,
        Config.CACHE_DIR,
        mode="test",
        transform=get_img_custom_transforms("val"),
    )

    # Loaders
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
