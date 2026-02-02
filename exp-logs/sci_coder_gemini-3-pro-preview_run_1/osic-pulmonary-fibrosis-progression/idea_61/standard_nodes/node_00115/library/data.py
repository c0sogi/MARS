import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config, seed_everything

# --------------------------------------------------------------------------
# Constants & Encoders
# --------------------------------------------------------------------------
LUNG_WINDOW_LEVEL = -600
LUNG_WINDOW_WIDTH = 1500


def get_lung_window(img):
    """Applies lung windowing to CT scan."""
    lower = LUNG_WINDOW_LEVEL - LUNG_WINDOW_WIDTH / 2
    upper = LUNG_WINDOW_LEVEL + LUNG_WINDOW_WIDTH / 2
    img = np.clip(img, lower, upper)
    # Normalize to 0-1 then to 0-255
    img = (img - lower) / (upper - lower)
    img = (img * 255).astype(np.uint8)
    return img


def load_scan(path):
    """
    Loads a CT scan from a directory of DICOM files.
    Returns a 3D numpy array (Z, Y, X) in Hounsfield Units.
    """
    try:
        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            return None

        slices = [pydicom.dcmread(s) for s in files]
        # Sort by ImagePositionPatient Z coordinate if available, else InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Stack slices
        image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

        # Convert to Hounsfield Units
        # Most scanners have slope/intercept in metadata
        if hasattr(slices[0], "RescaleIntercept") and hasattr(
            slices[0], "RescaleSlope"
        ):
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope
            image = image * slope + intercept

        return image
    except Exception as e:
        print(f"Error loading scan at {path}: {e}")
        return None


def generate_tri_slab(volume, axis=0, num_slabs=3, overlap=0.15):
    """
    Generates a 3-channel image by splitting the volume into overlapping slabs
    and computing Maximum Intensity Projection (MIP).

    Args:
        volume (np.array): 3D volume (Z, Y, X)
        axis (int): 0 for Axial (split Z), 1 for Coronal (split Y)
    """
    if volume is None:
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)

    # If generating Coronal from (Z, Y, X), we need to permute to put Y as primary axis
    # Axial (Z-split): Input (Z, Y, X) -> Slabs along Z -> MIP(Z) -> Result (Y, X)
    # Coronal (Y-split): Input (Z, Y, X) -> Permute to (Y, Z, X) -> Slabs along Y -> MIP(Y) -> Result (Z, X)

    if axis == 1:
        # Permute to (Y, Z, X)
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]
    if depth < num_slabs:
        # Handle edge case with very few slices by duplicating
        # Just return middle slice repeated or max proj of whole thing
        mip = np.max(volume, axis=0)
        mip = get_lung_window(mip)
        img = np.stack([mip] * 3, axis=-1)
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        return img

    # Calculate slab boundaries
    slab_depth = depth / num_slabs
    overlap_depth = depth * overlap

    channels = []

    for i in range(num_slabs):
        # Determine start and end indices
        # Core range: [i * slab_depth, (i+1) * slab_depth]
        # Add overlap
        start = max(0, int(i * slab_depth - overlap_depth / 2))
        end = min(depth, int((i + 1) * slab_depth + overlap_depth / 2))

        # Extract slab
        slab = volume[start:end, :, :]

        # MIP
        if slab.shape[0] > 0:
            mip = np.max(slab, axis=0)
        else:
            mip = np.zeros(volume.shape[1:], dtype=np.float32)

        # Windowing
        mip = get_lung_window(mip)
        channels.append(mip)

    # Stack to RGB
    img = np.stack(channels, axis=-1)  # (H, W, 3)

    # Resize
    img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

    return img


def cache_patient_images(patient_id, dicom_dir, cache_dir):
    """
    Checks cache, processes DICOM if needed, saves .npy.
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    if os.path.exists(axial_path) and os.path.exists(coronal_path):
        return

    # Load volume
    full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
    volume = load_scan(full_path)

    # Generate Axial (Axis 0)
    axial_img = generate_tri_slab(volume, axis=0)
    np.save(axial_path, axial_img)

    # Generate Coronal (Axis 1)
    coronal_img = generate_tri_slab(volume, axis=1)
    np.save(coronal_path, coronal_img)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, transform=None, mode="train"):
        self.df = df
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        # Tabular Normalization Constants (Approximate from EDA)
        self.age_mean = 67.0
        self.age_std = 7.0
        self.percent_mean = 77.0
        self.percent_std = 20.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # Handle missing cache (should not happen if setup is correct)
        if os.path.exists(axial_path):
            img_axial = np.load(axial_path)
        else:
            img_axial = np.zeros(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
            )

        if os.path.exists(coronal_path):
            img_coronal = np.load(coronal_path)
        else:
            img_coronal = np.zeros(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
            )

        # 2. Augmentations
        if self.transform:
            # Apply same spatial transform to both if possible, or independent
            # For simplicity and robustness, we apply independent spatial transforms
            # or use a fixed seed if we wanted them aligned.
            # Given they are orthogonal views, independent is acceptable/better for regularization.
            aug_ax = self.transform(image=img_axial)["image"]
            aug_cor = self.transform(image=img_coronal)["image"]
        else:
            # Fallback to ToTensor
            aug_ax = ToTensorV2()(image=img_axial)["image"]
            aug_cor = ToTensorV2()(image=img_coronal)["image"]

        # Normalize images to 0-1 float
        img_axial = aug_ax.float() / 255.0
        img_coronal = aug_cor.float() / 255.0

        # 3. Tabular Features
        # We use Baseline features for prediction context
        # Features: [Age, Sex_M, Sex_F, Smoking_Ex, Smoking_Never, Smoking_Current, Percent]

        age_norm = (row["Baseline_Age"] - self.age_mean) / self.age_std
        percent_norm = (row["Baseline_Percent"] - self.percent_mean) / self.percent_std

        # One-hot encoding
        sex = row["Baseline_Sex"]
        sex_m = 1.0 if sex == "Male" else 0.0
        sex_f = 1.0 if sex == "Female" else 0.0

        smoke = row["Baseline_SmokingStatus"]
        smoke_ex = 1.0 if smoke == "Ex-smoker" else 0.0
        smoke_never = 1.0 if smoke == "Never smoked" else 0.0
        smoke_curr = 1.0 if smoke == "Currently smokes" else 0.0

        tabular = torch.tensor(
            [age_norm, sex_m, sex_f, smoke_ex, smoke_never, smoke_curr, percent_norm],
            dtype=torch.float32,
        )

        # 4. Meta Data for Reconstruction
        # Model needs: Baseline_FVC and Week_Diff to predict current FVC
        baseline_fvc = float(row["Baseline_FVC"])
        week_diff = float(row["Week_Diff"])

        meta = torch.tensor([baseline_fvc, week_diff], dtype=torch.float32)

        # 5. Target
        if self.mode != "test":
            target_fvc = float(row["FVC"])
            # Dummy confidence for training loader consistency
            target = torch.tensor([target_fvc], dtype=torch.float32)
        else:
            target = torch.tensor([0.0], dtype=torch.float32)

        return {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "tabular": tabular,
            "meta": meta,
            "target": target,
            "patient_week": row.get("Patient_Week", ""),
        }


# --------------------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------------------
def prepare_train_dataframe(csv_path):
    """
    Prepares training data by identifying baseline for each patient
    and calculating week differences.
    """
    df = pd.read_csv(csv_path)

    # Identify baseline row for each patient (min Weeks)
    # We sort by Weeks and take the first one
    baseline_df = (
        df.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
    )

    # Select relevant baseline columns
    baseline_cols = [
        "Patient",
        "FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Weeks",
    ]
    baseline_df = baseline_df[baseline_cols]

    # Rename for merge
    baseline_df.columns = [
        "Patient",
        "Baseline_FVC",
        "Baseline_Percent",
        "Baseline_Age",
        "Baseline_Sex",
        "Baseline_SmokingStatus",
        "Baseline_Week",
    ]

    # Merge back to original df
    merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

    # Calculate Week Diff
    merged_df["Week_Diff"] = merged_df["Weeks"] - merged_df["Baseline_Week"]

    return merged_df


def prepare_test_dataframe(csv_path):
    """
    Prepares test data. Metadata/test.csv already has Baseline info merged.
    """
    df = pd.read_csv(csv_path)

    # Calculate Week Diff
    df["Week_Diff"] = df["Predict_Week"] - df["Baseline_Week"]

    return df


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare data and return dataloaders.
    """
    # 1. Prepare DataFrames
    train_df = prepare_train_dataframe(Config.TRAIN_CSV)
    val_df = prepare_train_dataframe(Config.VAL_CSV)
    test_df = prepare_test_dataframe(Config.TEST_CSV)

    if Config.DEBUG:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        test_df = test_df.head(20)

    # 2. Caching Images
    # Collect all unique patients and directories
    all_patients = pd.concat(
        [
            train_df[["Patient", "dicom_dir"]],
            val_df[["Patient", "dicom_dir"]],
            test_df[["Patient", "dicom_dir"]],
        ]
    ).drop_duplicates()

    print(f"Checking/Processing images for {len(all_patients)} patients...")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    for _, row in all_patients.iterrows():
        try:
            cache_patient_images(row["Patient"], row["dicom_dir"], Config.CACHE_DIR)
        except Exception as e:
            print(f"Failed to process {row['Patient']}: {e}")

    # 3. Create Datasets
    train_ds = OSICDataset(
        train_df, Config.CACHE_DIR, transform=get_transforms("train"), mode="train"
    )

    val_ds = OSICDataset(
        val_df, Config.CACHE_DIR, transform=get_transforms("val"), mode="val"
    )

    test_ds = OSICDataset(
        test_df, Config.CACHE_DIR, transform=get_transforms("test"), mode="test"
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
