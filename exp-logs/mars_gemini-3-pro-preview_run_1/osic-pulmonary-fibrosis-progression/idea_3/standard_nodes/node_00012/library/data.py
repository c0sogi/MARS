import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom

from library.config import Config
from library.utils import seed_everything

# ==========================================
# Helper Functions
# ==========================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specific mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )


def get_pixels_hu(slices):
    """
    Converts a list of dicom slices to a numpy array of Hounsfield Units.
    """
    # Stack the pixel arrays
    try:
        image = np.stack([s.pixel_array for s in slices])
    except Exception:
        # Fallback for inconsistent shapes
        return np.zeros((len(slices), 512, 512), dtype=np.int16)

    image = image.astype(np.int16)

    # Convert to Hounsfield Units (HU)
    for slice_number in range(len(slices)):
        try:
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
        except AttributeError:
            intercept = -1024
            slope = 1

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def load_dicom_volume(dicom_dir):
    """
    Loads a 3D volume from a directory of DICOM files.
    """
    if not os.path.exists(dicom_dir):
        return np.zeros((10, 512, 512), dtype=np.int16)

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, 512, 512), dtype=np.int16)

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return np.zeros((10, 512, 512), dtype=np.int16)

    # Sort by ImagePositionPatient Z coordinate
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback: sort by InstanceNumber
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass  # Keep original order

    return get_pixels_hu(slices)


def normalize_hu(image):
    """
    Clips HU to lung window [-1000, 400] and normalizes to [0, 1].
    """
    min_bound = -1000.0
    max_bound = 400.0

    image = (image - min_bound) / (max_bound - min_bound)
    image = np.clip(image, 0, 1)
    return image


def generate_dual_views(volume, img_size=224):
    """
    Generates Axial and Coronal Tri-Slab views from a 3D volume.

    Args:
        volume (np.array): 3D array (Z, Y, X) of HU values.
        img_size (int): Output spatial resolution.

    Returns:
        dict: {'axial': np.array (H, W, 3), 'coronal': np.array (H, W, 3)}
              Values are in [0, 1] range.
    """
    # Normalize HU first
    vol_norm = normalize_hu(volume)

    z, y, x = vol_norm.shape

    # --- Axial View (Top-down) ---
    # Split Z into 3 chunks
    chunk_size_z = max(1, z // 3)
    axial_channels = []
    for i in range(3):
        start = i * chunk_size_z
        end = (i + 1) * chunk_size_z if i < 2 else z

        if start >= z:
            slab = vol_norm[z - 1 : z, :, :]
        else:
            slab = vol_norm[start:end, :, :]

        # MIP along Z (axis 0)
        if slab.shape[0] > 0:
            mip = np.max(slab, axis=0)
        else:
            mip = np.zeros((y, x))
        axial_channels.append(mip)

    axial_img = np.stack(axial_channels, axis=-1)  # (Y, X, 3)
    axial_img = cv2.resize(axial_img, (img_size, img_size))

    # --- Coronal View (Front-back) ---
    # Split Y into 3 chunks
    chunk_size_y = max(1, y // 3)
    coronal_channels = []
    for i in range(3):
        start = i * chunk_size_y
        end = (i + 1) * chunk_size_y if i < 2 else y

        if start >= y:
            slab = vol_norm[:, y - 1 : y, :]
        else:
            slab = vol_norm[:, start:end, :]

        # MIP along Y (axis 1) -> Result is (Z, X)
        if slab.shape[1] > 0:
            mip = np.max(slab, axis=1)
        else:
            mip = np.zeros((z, x))
        coronal_channels.append(mip)

    coronal_img = np.stack(coronal_channels, axis=-1)  # (Z, X, 3)
    coronal_img = cv2.resize(coronal_img, (img_size, img_size))

    return {
        "axial": axial_img.astype(np.float32),
        "coronal": coronal_img.astype(np.float32),
    }


def process_patient_data(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Handles caching logic for patient image processing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception:
            pass

    # Process
    patient_dicom_dir = os.path.join(dicom_dir, patient_id)
    volume = load_dicom_volume(patient_dicom_dir)
    views = generate_dual_views(volume, Config.IMG_SIZE)

    # Save
    np.save(cache_path, views)

    return views


# ==========================================
# Dataset Class
# ==========================================


class LungDataset(Dataset):
    def __init__(
        self, df, dicom_dir, mode="train", transform=None, load_cached_data=True
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            dicom_dir (str): Root directory containing patient folders.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use disk caching.
        """
        self.df = df.copy()
        self.dicom_dir = dicom_dir
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Handle Debugging
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLES]

        # Feature Encoding Maps
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Build baseline lookup for training data
        self.patient_baseline_fvc = {}
        self.patient_baseline_week = {}

        if self.mode in ["train", "val"]:
            self._build_baseline_lookup()

    def _build_baseline_lookup(self):
        # Identify the baseline (earliest) visit for each patient
        for pid, group in self.df.groupby("Patient"):
            group = group.sort_values("Weeks")
            if not group.empty:
                base_row = group.iloc[0]
                self.patient_baseline_fvc[pid] = float(base_row["FVC"])
                self.patient_baseline_week[pid] = float(base_row["Weeks"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        views = process_patient_data(
            patient_id, self.dicom_dir, Config.CACHE_DIR, self.load_cached_data
        )

        axial = views["axial"]
        coronal = views["coronal"]

        # 2. Augmentations
        if self.transform:
            # Apply transforms independently to both views
            res_axial = self.transform(image=axial)
            axial_tensor = res_axial["image"]

            res_coronal = self.transform(image=coronal)
            coronal_tensor = res_coronal["image"]
        else:
            t = ToTensorV2()
            axial_tensor = t(image=axial)["image"]
            coronal_tensor = t(image=coronal)["image"]

        # 3. Tabular Features
        # Handle different column names in train vs test
        if "Age" in row:
            age = row["Age"]
            sex = row["Sex"]
            smoke = row["SmokingStatus"]
        else:
            age = row.get("Baseline_Age", 65)
            sex = row.get("Baseline_Sex", "Male")
            smoke = row.get("Baseline_SmokingStatus", "Ex-smoker")

        # Normalize Age
        age_norm = (float(age) - 50.0) / 50.0
        sex_enc = self.sex_map.get(sex, 0)
        smoke_enc = self.smoke_map.get(smoke, 0)

        # One-hot encoding for smoking
        smoke_ohe = [0, 0, 0]
        smoke_ohe[smoke_enc] = 1

        # Feature Vector: [Age, Sex, Smoke_0, Smoke_1, Smoke_2]
        tab_features = np.array([age_norm, sex_enc] + smoke_ohe, dtype=np.float32)

        # 4. Target and Meta
        meta = {}
        meta["patient_id"] = patient_id

        if self.mode in ["train", "val"]:
            target = float(row["FVC"])

            # Retrieve baseline info from lookup
            base_fvc = self.patient_baseline_fvc.get(patient_id, 2500.0)
            base_week = self.patient_baseline_week.get(patient_id, 0)

            current_week = float(row["Weeks"])
            time_delta = current_week - base_week

        else:
            # Test mode
            target = 0.0
            base_fvc = float(row.get("Baseline_FVC", 2500.0))
            base_week = float(row.get("Baseline_Week", 0))
            predict_week = float(row.get("Predict_Week", 0))
            time_delta = predict_week - base_week
            meta["patient_week"] = row.get("Patient_Week", "")

        return {
            "axial": axial_tensor,
            "coronal": coronal_tensor,
            "tabular": torch.tensor(tab_features, dtype=torch.float32),
            "time_delta": torch.tensor(time_delta, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "meta": meta,
        }


def get_dataloaders():
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Create Datasets
    train_ds = LungDataset(
        train_df,
        Config.DICOM_TRAIN_DIR,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    val_ds = LungDataset(
        val_df,
        Config.DICOM_TRAIN_DIR,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    test_ds = LungDataset(
        test_df,
        Config.DICOM_TEST_DIR,
        mode="test",
        transform=get_transforms("test"),
        load_cached_data=True,
    )

    # Create Loaders
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

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
