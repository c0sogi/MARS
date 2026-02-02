import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Attempt to import pydicom, handle absence gracefully
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def load_scan(path):
    """
    Loads DICOM scans from a directory, sorts them, and stacks into a 3D volume.
    Falls back to raw binary reading if pydicom is not available.
    """
    if not os.path.exists(path):
        return np.zeros((1, 512, 512), dtype=np.float32)

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((1, 512, 512), dtype=np.float32)

    # Sort by numeric instance number inferred from filename
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()

    slices = []
    for f in files:
        file_path = os.path.join(path, f)

        if HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(file_path)
                img = ds.pixel_array.astype(np.float32)

                # Apply Rescale Slope/Intercept to get HU
                intercept = getattr(ds, "RescaleIntercept", -1024)
                slope = getattr(ds, "RescaleSlope", 1)
                img = img * slope + intercept
                slices.append(img)
            except Exception:
                continue
        else:
            # Fallback: Raw binary read for 512x512 images
            try:
                H, W = 512, 512
                n_pixels = H * W
                n_bytes = n_pixels * 2  # uint16

                file_size = os.path.getsize(file_path)
                offset = file_size - n_bytes

                if offset < 0:
                    continue

                with open(file_path, "rb") as f_obj:
                    f_obj.seek(offset)
                    img_raw = np.frombuffer(f_obj.read(n_bytes), dtype=np.uint16)

                if img_raw.size != n_pixels:
                    continue

                img = img_raw.reshape((H, W)).astype(np.float32)
                # Approximate HU conversion for standard CT
                img = img * 1.0 - 1024.0
                slices.append(img)
            except Exception:
                continue

    if not slices:
        return np.zeros((1, 512, 512), dtype=np.float32)

    # Stack along Z axis -> (Depth, Height, Width)
    volume = np.stack(slices, axis=0)
    return volume


def generate_tri_slabs(volume):
    """
    Generates Axial and Coronal RGB images using overlapping slabs.
    Input Volume: (Z, Y, X)
    """
    # 1. Normalize to Lung Window [-1000, 400] -> [0, 1]
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400.0

    def get_projection(vol_data):
        # vol_data shape: (Depth, H, W)
        depth = vol_data.shape[0]
        if depth < 1:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Define overlapping slabs: 0-40%, 30-70%, 60-100%
        starts = [0.0, 0.3, 0.6]
        ends = [0.4, 0.7, 1.0]

        channels = []
        for s, e in zip(starts, ends):
            idx_start = int(s * depth)
            idx_end = int(e * depth)
            # Ensure valid range
            idx_end = max(idx_end, idx_start + 1)
            idx_end = min(idx_end, depth)

            slab = vol_data[idx_start:idx_end, :, :]
            if slab.shape[0] > 0:
                mip = np.max(slab, axis=0)
            else:
                mip = np.zeros((vol_data.shape[1], vol_data.shape[2]), dtype=np.float32)
            channels.append(mip)

        # Stack to RGB (H, W, 3)
        img = np.stack(channels, axis=-1)

        # Resize to target size
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        return img

    # Axial View: Projection along Z (axis 0)
    axial_img = get_projection(volume)

    # Coronal View: Projection along Y (axis 1)
    # Transpose to make Y the depth axis: (Y, Z, X)
    vol_coronal = np.transpose(volume, (1, 0, 2))
    coronal_img = get_projection(vol_coronal)

    return axial_img, coronal_img


def process_patient_images(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Manages caching of processed images.
    """
    os.makedirs(cache_dir, exist_ok=True)

    path_axial = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    path_coronal = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(path_axial) and os.path.exists(path_coronal):
        try:
            axial = np.load(path_axial)
            coronal = np.load(path_coronal)
            return axial, coronal
        except Exception:
            pass  # Fallback to re-processing

    # Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
    volume = load_scan(full_path)
    axial, coronal = generate_tri_slabs(volume)

    # Save to cache
    np.save(path_axial, axial)
    np.save(path_coronal, coronal)

    return axial, coronal


class LungDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train", load_cached_data=True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        dicom_dir = row["dicom_dir"]
        axial, coronal = process_patient_images(
            patient_id, dicom_dir, Config.CACHE_DIR, self.load_cached_data
        )

        # 2. Apply Transforms
        if self.transforms:
            # Apply independently as views are not spatially aligned
            aug_ax = self.transforms(image=axial)["image"]
            aug_cor = self.transforms(image=coronal)["image"]
        else:
            aug_ax = torch.tensor(axial.transpose(2, 0, 1), dtype=torch.float32)
            aug_cor = torch.tensor(coronal.transpose(2, 0, 1), dtype=torch.float32)

        # 3. Process Tabular Data
        # Normalize Age (approx mean 50, range 50)
        age_val = float(row["Baseline_Age"] if "Baseline_Age" in row else row["Age"])
        age = (age_val - 50.0) / 50.0

        # Encode Sex
        sex_val = row["Baseline_Sex"] if "Baseline_Sex" in row else row["Sex"]
        sex = self.sex_map.get(sex_val, 0)

        # Encode Smoking
        smoke_val = (
            row["Baseline_SmokingStatus"]
            if "Baseline_SmokingStatus" in row
            else row["SmokingStatus"]
        )
        smoke = self.smoke_map.get(smoke_val, 0)

        # Normalize Percent (approx mean 80, range 20)
        pct_val = float(
            row["Baseline_Percent"] if "Baseline_Percent" in row else row["Percent"]
        )
        percent = (pct_val - 80.0) / 20.0

        # Tabular Vector: [Age, Sex, Smoke, Percent]
        tabular = torch.tensor(
            [age, float(sex), float(smoke), percent], dtype=torch.float32
        )

        # 4. Meta & Target
        # Meta: Relative Week and Baseline FVC for parametric head
        week = float(row["Predict_Week"] if "Predict_Week" in row else row["Weeks"])
        base_week = float(row["Baseline_Week"]) if "Baseline_Week" in row else 0.0
        base_fvc = float(row["Baseline_FVC"]) if "Baseline_FVC" in row else 0.0

        rel_week = week - base_week
        meta_tensor = torch.tensor([rel_week, base_fvc], dtype=torch.float32)

        if self.mode in ["train", "val"]:
            target = float(row["FVC"])
            return {
                "axial": aug_ax,
                "coronal": aug_cor,
                "tabular": tabular,
                "meta": meta_tensor,
                "target": torch.tensor(target, dtype=torch.float32),
            }
        else:
            return {
                "axial": aug_ax,
                "coronal": aug_cor,
                "tabular": tabular,
                "meta": meta_tensor,
                "patient_week": row["Patient_Week"],
            }


def get_dataloaders(debug=False):
    """
    Creates Train and Validation DataLoaders.
    """
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Augmentation for Training
    train_transform = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    # Preprocessing for Validation
    val_transform = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    train_ds = LungDataset(
        train_df, transforms=train_transform, mode="train", load_cached_data=True
    )
    val_ds = LungDataset(
        val_df, transforms=val_transform, mode="val", load_cached_data=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates Test DataLoader.
    """
    test_df = pd.read_csv(Config.TEST_CSV)

    test_transform = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    test_ds = LungDataset(
        test_df, transforms=test_transform, mode="test", load_cached_data=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
