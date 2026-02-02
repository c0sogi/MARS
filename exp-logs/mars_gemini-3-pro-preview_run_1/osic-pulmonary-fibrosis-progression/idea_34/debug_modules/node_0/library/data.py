import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything

# ==========================================
# 1. Helper Functions
# ==========================================


def get_img_path(dicom_dir, patient_id):
    """Constructs the full path to the DICOM directory."""
    return os.path.join(Config.INPUT_DIR, dicom_dir)


def load_scan(path):
    """
    Loads a CT scan from a directory of DICOM files.
    Returns a 3D numpy array (Z, Y, X) in Hounsfield Units.
    """
    try:
        if not os.path.exists(path):
            return None

        slices = [
            pydicom.dcmread(os.path.join(path, s))
            for s in os.listdir(path)
            if s.endswith(".dcm")
        ]
        if not slices:
            return None

        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Extract images and handle slope/intercept
        images = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", 0)
            img = slope * img + intercept
            images.append(img)

        volume = np.stack(images)
        return volume
    except Exception as e:
        # Return None if loading fails; dataset will handle this by generating a blank volume
        return None


def generate_tri_slab(volume, view="axial"):
    """
    Generates a 3-channel image using overlapping Maximum Intensity Projections (MIP).

    Args:
        volume: 3D numpy array (D, H, W)
        view: 'axial' (Z-axis) or 'coronal' (Y-axis)

    Returns:
        img: (H_out, W_out, 3) numpy array, normalized to [0, 255]
    """
    if volume is None or volume.size == 0:
        # Return blank image if volume is invalid
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    # Standard Lung Windowing [-1000, 400]
    volume = np.clip(volume, -1000, 400)

    # Normalize to [0, 1] for processing
    volume = (volume - (-1000)) / (400 - (-1000))

    # Determine slicing axis
    if view == "axial":
        # Axial is usually axis 0 (Z) in (Z, Y, X)
        # We slice along Z, project onto Y-X
        target_axis = 0
    elif view == "coronal":
        # Coronal is usually axis 1 (Y) in (Z, Y, X)
        # We slice along Y, project onto Z-X
        # To make it easier, we transpose to make Y the first axis
        volume = np.transpose(volume, (1, 0, 2))
        target_axis = 0
    else:
        raise ValueError(f"Unknown view: {view}")

    depth = volume.shape[0]

    # Define slab boundaries with overlap
    # Points: 0, 0.33, 0.66, 1.0
    p1 = 0
    p2 = int(depth * 0.33)
    p3 = int(depth * 0.66)
    p4 = depth

    overlap = int(depth * Config.SLAB_OVERLAP)

    # Indices with clamping
    idx1_start, idx1_end = 0, min(p2 + overlap, depth)
    idx2_start, idx2_end = max(0, p2 - overlap), min(p3 + overlap, depth)
    idx3_start, idx3_end = max(0, p3 - overlap), depth

    # Extract slabs
    slab1 = volume[idx1_start:idx1_end, :, :]
    slab2 = volume[idx2_start:idx2_end, :, :]
    slab3 = volume[idx3_start:idx3_end, :, :]

    # MIP (Maximum Intensity Projection)
    # Handle edge case where a slab might be empty (e.g. very small volume)
    def get_mip(slab):
        if slab.shape[0] == 0:
            return np.zeros(slab.shape[1:], dtype=np.float32)
        return np.max(slab, axis=0)

    c1 = get_mip(slab1)
    c2 = get_mip(slab2)
    c3 = get_mip(slab3)

    # Stack to 3 channels (H, W, 3)
    img = np.stack([c1, c2, c3], axis=-1)

    # Resize to target resolution
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    # Convert to uint8 [0, 255]
    img = (img * 255).astype(np.uint8)

    return img


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Strictly spatial only - no intensity/color jitter.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def prepare_dataframe(df_path, mode="train"):
    """
    Loads and preprocesses the metadata dataframe.
    Extracts baseline features for train/val sets.
    """
    df = pd.read_csv(df_path)

    if mode in ["train", "val"]:
        # We need to identify baseline stats (Week ~ 0) for each patient
        # and broadcast them to all rows for that patient.

        # Calculate absolute week to find closest to 0
        df["abs_week"] = df["Weeks"].abs()

        # Sort by patient and abs_week
        df_sorted = df.sort_values(["Patient", "abs_week"])

        # Drop duplicates to keep first (closest to 0) as baseline
        baseline_df = df_sorted.drop_duplicates("Patient", keep="first")[
            ["Patient", "FVC", "Percent"]
        ]
        baseline_df = baseline_df.rename(
            columns={"FVC": "Baseline_FVC", "Percent": "Baseline_Percent"}
        )

        # Merge baseline info back to original df
        df = df.merge(baseline_df, on="Patient", how="left")

    elif mode == "test":
        # Test CSV already has Baseline_FVC, Baseline_Percent, etc.
        # We need to ensure column names align.
        # Test metadata has 'Predict_Week', we map it to 'Weeks' for consistency
        if "Predict_Week" in df.columns:
            df["Weeks"] = df["Predict_Week"]

    return df


# ==========================================
# 2. Dataset Class
# ==========================================


class LungDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train", load_cached_data=True):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def _get_image(self, patient_id, dicom_dir, view):
        """
        Retrieves image from cache or processes from scratch.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_{view}.npy")

        # 1. Try Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                img = np.load(cache_path)
                return img
            except:
                pass  # Fallback to processing if load fails

        # 2. Process from DICOM
        full_path = get_img_path(dicom_dir, patient_id)
        volume = load_scan(full_path)
        img = generate_tri_slab(volume, view)

        # 3. Save to Cache
        try:
            np.save(cache_path, img)
        except:
            pass  # Ignore save errors

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]
        dicom_dir = row["dicom_dir"]

        # 1. Load Images
        img_axial = self._get_image(patient_id, dicom_dir, "axial")
        img_coronal = self._get_image(patient_id, dicom_dir, "coronal")

        # 2. Apply Transforms
        if self.transforms:
            # Albumentations expects dict
            res_ax = self.transforms(image=img_axial)
            img_axial = res_ax["image"]

            res_cor = self.transforms(image=img_coronal)
            img_coronal = res_cor["image"]

        # 3. Extract Tabular Features
        # Context for GLU: [Age, Sex, Smoking, Baseline_Percent]
        # Head Inputs: [Age, Sex, Smoking, Baseline_Percent] (Same, but used differently in model)

        # Use Baseline values if available (Test set has Baseline_Age, etc.)
        # Train set has Age (which is roughly baseline).

        age = row["Baseline_Age"] if "Baseline_Age" in row else row["Age"]
        sex = row["Baseline_Sex"] if "Baseline_Sex" in row else row["Sex"]
        smoke = (
            row["Baseline_SmokingStatus"]
            if "Baseline_SmokingStatus" in row
            else row["SmokingStatus"]
        )
        percent = row["Baseline_Percent"]

        # Encode
        sex_enc = self.sex_map.get(sex, 0)
        smoke_enc = self.smoke_map.get(smoke, 1)

        # Create feature vector (Raw values as requested)
        # [Age, Sex, Smoking, Percent]
        static_features = torch.tensor(
            [float(age), float(sex_enc), float(smoke_enc), float(percent)],
            dtype=torch.float32,
        )

        # 4. Dynamic & Target Data
        baseline_fvc = float(row["Baseline_FVC"])

        # In test set, row['Weeks'] is the target week. In train, it's the visit week.
        # We need relative week from baseline.
        # Test set: Weeks is already relative or absolute?
        # Metadata generation says: Test Weeks = Predict_Week. Baseline_Week is separate.
        # We need (Week - Baseline_Week).

        current_week = float(row["Weeks"])
        baseline_week = float(row["Baseline_Week"]) if "Baseline_Week" in row else 0.0
        # Note: In train set, Weeks is already relative to baseline (where baseline is ~0).
        # But to be safe, if we calculated baseline from a specific row, we should subtract that row's week.
        # For simplicity in train set, we assume Weeks is relative.
        # For test set, we explicitly compute relative.

        if self.mode == "test":
            relative_week = current_week - baseline_week
        else:
            relative_week = current_week  # Train weeks are already relative

        data = {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "static_features": static_features,
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "week": torch.tensor(relative_week, dtype=torch.float32),
        }

        if self.mode != "test":
            target = float(row["FVC"])
            data["target"] = torch.tensor(target, dtype=torch.float32)

        return data


# ==========================================
# 3. Data Loaders
# ==========================================


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=False,
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    # Load Dataframes
    train_df = prepare_dataframe(Config.TRAIN_CSV, mode="train")
    val_df = prepare_dataframe(Config.VAL_CSV, mode="val")
    test_df = prepare_dataframe(Config.TEST_CSV, mode="test")

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Datasets
    train_ds = LungDataset(
        train_df,
        transforms=get_transforms("train"),
        mode="train",
        load_cached_data=load_cached_data,
    )

    val_ds = LungDataset(
        val_df,
        transforms=get_transforms("val"),
        mode="val",
        load_cached_data=load_cached_data,
    )

    test_ds = LungDataset(
        test_df,
        transforms=get_transforms("test"),
        mode="test",
        load_cached_data=load_cached_data,
    )

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
