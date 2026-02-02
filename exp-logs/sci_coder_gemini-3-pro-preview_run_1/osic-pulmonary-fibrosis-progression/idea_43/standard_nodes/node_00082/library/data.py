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

# ==========================================
# 1. Helper Functions for DICOM & Image Proc
# ==========================================


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by InstanceNumber.
    """
    slices = []
    if not os.path.exists(path):
        return []

    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception:
                continue

    if not slices:
        return []

    # Sort by ImagePositionPatient Z if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts DICOM slices to a 3D numpy array of Hounsfield Units (HU).
    Handles slope/intercept and padding.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    for n, s in enumerate(slices):
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else -1024
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1

        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)

        image[n] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def normalize_hu(image):
    """
    Applies Lung Window and normalizes to [0, 255].
    Window Level: -600, Window Width: 1500
    Range: [-1350, 150]
    """
    min_hu = -1000
    max_hu = 400

    image = np.clip(image, min_hu, max_hu)

    # Normalize to 0-1
    image = (image - min_hu) / (max_hu - min_hu)

    # Scale to 0-255 for standard image processing
    image = (image * 255).astype(np.uint8)
    return image


def generate_tri_slab(volume, axis_idx, overlap=0.15):
    """
    Generates a 3-channel Tri-Slab image from a 3D volume via MIP.

    Args:
        volume: 3D numpy array (Z, Y, X)
        axis_idx: Axis to split along (0 for Axial/Z, 1 for Coronal/Y)
        overlap: Fraction of total depth to overlap between slabs

    Returns:
        np.array: (H, W, 3) image
    """
    # Dimensions
    depth = volume.shape[axis_idx]

    # If depth is too small, just repeat the max projection
    if depth < 3:
        if axis_idx == 0:
            proj = np.max(volume, axis=0)
        else:
            proj = np.max(volume, axis=1)
        return np.stack([proj, proj, proj], axis=-1)

    # Define slab boundaries with overlap
    # We divide the range [0, depth] into 3 overlapping segments
    # Base segment size
    seg_size = depth / 3.0
    margin = depth * overlap

    # Slab 1: 0 to 1/3 + margin
    s1_start = 0
    s1_end = int(seg_size + margin)

    # Slab 2: 1/3 - margin to 2/3 + margin
    s2_start = int(seg_size - margin)
    s2_end = int(2 * seg_size + margin)

    # Slab 3: 2/3 - margin to end
    s3_start = int(2 * seg_size - margin)
    s3_end = depth

    # Ensure indices are valid
    s1_end = min(s1_end, depth)
    s2_start = max(0, s2_start)
    s2_end = min(s2_end, depth)
    s3_start = max(0, s3_start)

    # Extract slabs
    if axis_idx == 0:  # Axial split (along Z)
        slab1 = volume[s1_start:s1_end, :, :]
        slab2 = volume[s2_start:s2_end, :, :]
        slab3 = volume[s3_start:s3_end, :, :]

        # MIP along Z
        mip1 = np.max(slab1, axis=0)
        mip2 = np.max(slab2, axis=0)
        mip3 = np.max(slab3, axis=0)

    elif axis_idx == 1:  # Coronal split (along Y)
        slab1 = volume[:, s1_start:s1_end, :]
        slab2 = volume[:, s2_start:s2_end, :]
        slab3 = volume[:, s3_start:s3_end, :]

        # MIP along Y
        mip1 = np.max(slab1, axis=1)
        mip2 = np.max(slab2, axis=1)
        mip3 = np.max(slab3, axis=1)

    else:
        raise ValueError("Axis must be 0 (Axial) or 1 (Coronal)")

    # Stack to channels (H, W, 3)
    img = np.stack([mip1, mip2, mip3], axis=-1)
    return img


def resize_image(img, size=224):
    """Resizes image to target square resolution."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ==========================================
# 2. Data Preparation & Dataset Class
# ==========================================


def prepare_train_dataframe(df):
    """
    Prepares the training dataframe by identifying baseline characteristics.
    For each patient, the visit closest to Week 0 is considered baseline.
    """
    # Identify baseline row for each patient (min absolute weeks)
    df["Abs_Weeks"] = df["Weeks"].abs()
    baseline_indices = df.groupby("Patient")["Abs_Weeks"].idxmin()

    baseline_df = df.loc[
        baseline_indices,
        ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"],
    ]
    baseline_df = baseline_df.rename(
        columns={
            "Weeks": "Baseline_Week",
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Age": "Baseline_Age",
            "Sex": "Baseline_Sex",
            "SmokingStatus": "Baseline_SmokingStatus",
        }
    )

    # Merge baseline info back to the main dataframe
    merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

    # Calculate Delta Week (Time since baseline)
    merged_df["Delta_Week"] = merged_df["Weeks"] - merged_df["Baseline_Week"]

    return merged_df


class CTDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, load_cached_data=True):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Tabular Encoders
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Image Loading (with Caching)
        axial_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

        axial_img = None
        coronal_img = None

        # Try loading from cache
        if (
            self.load_cached_data
            and os.path.exists(axial_path)
            and os.path.exists(coronal_path)
        ):
            try:
                axial_img = np.load(axial_path)
                coronal_img = np.load(coronal_path)
            except Exception:
                pass  # Fallback to generation

        # Generate if not loaded
        if axial_img is None or coronal_img is None:
            dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
            slices = load_scan(dicom_dir)

            success = False
            if slices:
                try:
                    vol_hu = get_pixels_hu(slices)
                    vol_norm = normalize_hu(vol_hu)

                    # Generate Tri-Slabs
                    # Axial (Axis 0)
                    axial_raw = generate_tri_slab(
                        vol_norm, axis_idx=0, overlap=Config.SLAB_OVERLAP
                    )
                    axial_img = resize_image(axial_raw, Config.IMG_SIZE)

                    # Coronal (Axis 1)
                    coronal_raw = generate_tri_slab(
                        vol_norm, axis_idx=1, overlap=Config.SLAB_OVERLAP
                    )
                    coronal_img = resize_image(coronal_raw, Config.IMG_SIZE)
                    success = True
                except (RuntimeError, Exception):
                    # Cite debug_lesson_2: Handle missing codec dependencies gracefully
                    pass

            if not success:
                # Fallback for missing data or failed decompression: black image
                axial_img = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
                )
                coronal_img = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
                )

            # Cite debug_lesson_8: Cache fallback results to prevent repeated failure overheads
            np.save(axial_path, axial_img)
            np.save(coronal_path, coronal_img)

        # 2. Augmentation
        if self.transform:
            # Apply same spatial transform type, but independent randomness is acceptable/beneficial for dual views
            # Or we can apply transforms independently.
            aug_ax = self.transform(image=axial_img)["image"]
            aug_cor = self.transform(image=coronal_img)["image"]
        else:
            # Basic ToTensor
            base_t = ToTensorV2()
            aug_ax = base_t(image=axial_img)["image"]
            aug_cor = base_t(image=coronal_img)["image"]

        # Normalize images to [0, 1] float tensors
        img_axial = aug_ax.float() / 255.0
        img_coronal = aug_cor.float() / 255.0

        # 3. Tabular Data Processing
        # We use Baseline features for the network input

        # Normalize Numerical Features
        stats = Config.STATS

        # Note: For 'Weeks', we use the delta for prediction logic, but for network input
        # we strictly exclude time. We include Percent, Age.

        # Baseline Percent
        p_mean, p_std = stats["Percent"]["mean"], stats["Percent"]["std"]
        norm_percent = (row["Baseline_Percent"] - p_mean) / p_std

        # Baseline Age
        a_mean, a_std = stats["Age"]["mean"], stats["Age"]["std"]
        norm_age = (row["Baseline_Age"] - a_mean) / a_std

        # Categorical Encoding
        sex_code = self.sex_map.get(row["Baseline_Sex"], 0)
        smoke_code = self.smoke_map.get(row["Baseline_SmokingStatus"], 1)

        # One-hot encoding manually for dense vector
        # Sex (2 dim), Smoking (3 dim)
        sex_vec = [0, 0]
        sex_vec[sex_code] = 1

        smoke_vec = [0, 0, 0]
        smoke_vec[smoke_code] = 1

        # Construct Feature Vector
        # [Percent, Age, Sex_0, Sex_1, Smoke_0, Smoke_1, Smoke_2]
        # Total dims: 1 + 1 + 2 + 3 = 7
        tabular_features = np.array(
            [norm_percent, norm_age] + sex_vec + smoke_vec, dtype=np.float32
        )

        # 4. Target & Meta
        data = {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "tabular": torch.tensor(tabular_features, dtype=torch.float32),
            "delta_week": torch.tensor(
                row["Weeks"] - row["Baseline_Week"], dtype=torch.float32
            ),
            "baseline_fvc": torch.tensor(row["Baseline_FVC"], dtype=torch.float32),
            "patient_week": str(
                row.get("Patient_Week", f"{patient_id}_{row['Weeks']}")
            ),
        }

        if self.mode != "test":
            data["target"] = torch.tensor(row["FVC"], dtype=torch.float32)

        return data


# ==========================================
# 3. Transforms & Dataloaders
# ==========================================


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    # Load Metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # Debug Mode
    if debug:
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)

    # Prepare Dataframes (Compute Baselines)
    train_df = prepare_train_dataframe(train_df)
    val_df = prepare_train_dataframe(val_df)

    # Datasets
    train_ds = CTDataset(
        train_df,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=load_cached_data,
    )

    val_ds = CTDataset(
        val_df,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
    )

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Test DF already has Baseline_ columns from metadata generation
    # We just need to ensure column names match what CTDataset expects
    # In metadata/test.csv: Baseline_Week, Baseline_FVC, etc. are present.
    # 'Weeks' column is missing, but we have 'Predict_Week'.
    # We map 'Predict_Week' to 'Weeks' for the dataset logic.

    test_df = test_df.rename(columns={"Predict_Week": "Weeks"})

    test_ds = CTDataset(
        test_df,
        mode="test",
        transform=get_transforms("test"),
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
