import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import Config
from library.utils import seed_everything

# ==========================================
# 1. Helper Functions for Image Processing
# ==========================================


def load_dicom_volume(dicom_dir):
    """
    Loads a DICOM directory into a 3D numpy array (D, H, W).
    Converts to Hounsfield Units and applies Lung Window.
    """
    full_path = os.path.join(Config.INPUT_ROOT, dicom_dir)
    if not os.path.exists(full_path):
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Read DICOMs
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(full_path, f))
            slices.append(ds)
        except:
            continue

    if not slices:
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Sort by ImagePositionPatient Z (index 2) if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass  # Keep original order if sorting fails

    # Extract pixel data and convert to HU
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)

        # Intercept and Slope
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            img = slope * img.astype(np.float64)
            img = img.astype(np.float32)

        img += intercept
        images.append(img)

    volume = np.stack(images)  # (D, H, W)

    # Lung Window: Level -600, Width 1500 -> Range: [-1350, 150]
    # We clip to this range and normalize to [0, 1]
    L, W = -600, 1500
    lower = L - W // 2
    upper = L + W // 2

    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)

    return volume


def generate_tri_slabs(volume):
    """
    Generates Axial and Coronal Tri-Slab images from a 3D volume.
    Returns: (Axial_Img, Coronal_Img) as (224, 224, 3) numpy arrays.
    """
    D, H, W = volume.shape

    # Resize spatial dims to target size (224x224)
    # We resize slices individually to save memory
    if H != Config.IMG_SIZE or W != Config.IMG_SIZE:
        resized_slices = []
        for i in range(D):
            resized_slices.append(
                cv2.resize(volume[i], (Config.IMG_SIZE, Config.IMG_SIZE))
            )
        volume = np.stack(resized_slices)  # (D, 224, 224)

    # --- Axial Processing ---
    # Split D into 3 overlapping slabs: 0-40%, 30-70%, 60-100% (approx 15% overlap)
    ranges = [(0.0, 0.40), (0.30, 0.70), (0.60, 1.00)]  # Slab 1  # Slab 2  # Slab 3

    axial_channels = []
    for start_frac, end_frac in ranges:
        start_idx = int(D * start_frac)
        end_idx = int(D * end_frac)

        # Ensure valid bounds
        start_idx = max(0, start_idx)
        end_idx = min(D, max(start_idx + 1, end_idx))

        slab = volume[start_idx:end_idx, :, :]
        if slab.shape[0] > 0:
            mip = np.max(slab, axis=0)
        else:
            mip = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
        axial_channels.append(mip)

    axial_img = np.stack(axial_channels, axis=-1)  # (224, 224, 3)

    # --- Coronal Processing ---
    # Transpose to (Y, Z, X) -> (H, D, W)
    coronal_vol = volume.transpose(1, 0, 2)
    Y_dim = coronal_vol.shape[0]

    # Resize the slices (Z, X) to 224x224.
    # Input slice is (D, 224). We resize to (224, 224).
    resized_cor_slices = []
    for i in range(Y_dim):
        resized_cor_slices.append(
            cv2.resize(coronal_vol[i], (Config.IMG_SIZE, Config.IMG_SIZE))
        )
    coronal_vol = np.stack(resized_cor_slices)  # (Y, 224, 224)

    coronal_channels = []
    for start_frac, end_frac in ranges:
        start_idx = int(Y_dim * start_frac)
        end_idx = int(Y_dim * end_frac)

        start_idx = max(0, start_idx)
        end_idx = min(Y_dim, max(start_idx + 1, end_idx))

        slab = coronal_vol[start_idx:end_idx, :, :]
        if slab.shape[0] > 0:
            mip = np.max(slab, axis=0)
        else:
            mip = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
        coronal_channels.append(mip)

    coronal_img = np.stack(coronal_channels, axis=-1)

    return axial_img, coronal_img


def get_cached_images(patient_id, dicom_dir, load_cached_data=True):
    """
    Retrieves cached images or generates them.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    ax_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
    cor_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

    if load_cached_data and os.path.exists(ax_path) and os.path.exists(cor_path):
        try:
            ax = np.load(ax_path)
            cor = np.load(cor_path)
            return ax, cor
        except:
            pass

    # Generate
    vol = load_dicom_volume(dicom_dir)
    ax, cor = generate_tri_slabs(vol)

    # Save
    np.save(ax_path, ax)
    np.save(cor_path, cor)

    return ax, cor


# ==========================================
# 2. Tabular Preprocessing
# ==========================================


class TabularPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe_sex = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe_smoke = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def fit(self, df):
        # Fit on numerical columns: Weeks, Baseline_Percent, Baseline_Age
        num_data = df[["Weeks", "Baseline_Percent", "Baseline_Age"]].values
        self.scaler.fit(num_data)

        # Fit on categorical
        self.ohe_sex.fit(df[["Baseline_Sex"]].values)
        self.ohe_smoke.fit(df[["Baseline_SmokingStatus"]].values)

    def transform(self, df):
        num_data = df[["Weeks", "Baseline_Percent", "Baseline_Age"]].values
        num_scaled = self.scaler.transform(num_data)

        sex_enc = self.ohe_sex.transform(df[["Baseline_Sex"]].values)
        smoke_enc = self.ohe_smoke.transform(df[["Baseline_SmokingStatus"]].values)

        # Concatenate
        return np.hstack([num_scaled, sex_enc, smoke_enc]).astype(np.float32)


# ==========================================
# 3. Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(self, df, preprocessor, mode="train", load_cache=True):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.mode = mode
        self.load_cache = load_cache

        # Pre-compute tabular features
        self.tabular_features = self.preprocessor.transform(self.df)

        # ImageNet Normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD
                ),
            ]
        )

        # Augmentation (Spatial Only) for training
        if self.mode == "train":
            self.aug = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                ]
            )
        else:
            self.aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]
        dicom_dir = row["dicom_dir"]

        # 1. Images
        ax_np, cor_np = get_cached_images(pid, dicom_dir, self.load_cache)

        # Apply Normalization
        ax_t = self.normalize(ax_np)
        cor_t = self.normalize(cor_np)

        # Apply Augmentation
        if self.aug:
            ax_t = self.aug(ax_t)
            cor_t = self.aug(cor_t)

        # 2. Tabular
        tab_vec = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # 3. Targets and Metadata
        fvc = float(row["FVC"]) if "FVC" in row else 0.0
        base_fvc = float(row["Baseline_FVC"])
        base_week = float(row["Baseline_Week"]) if "Baseline_Week" in row else 0.0
        week = float(row["Weeks"])

        return {
            "axial": ax_t,
            "coronal": cor_t,
            "tabular": tab_vec,
            "fvc": torch.tensor(fvc, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "week": torch.tensor(week, dtype=torch.float32),
            "base_week": torch.tensor(base_week, dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"] if "Patient_Week" in row else f"{pid}_{int(week)}"
            ),
        }


# ==========================================
# 4. Data Preparation Logic
# ==========================================


def prepare_dataframes():
    """
    Loads metadata and prepares train/val/test dataframes with Baseline columns.
    """
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Helper to add baseline info to train/val
    def add_baseline_info(df):
        # Find baseline rows (min absolute weeks)
        df["abs_weeks"] = df["Weeks"].abs()
        idx = df.groupby("Patient")["abs_weeks"].idxmin()
        baseline_df = df.loc[
            idx, ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus", "Weeks"]
        ]

        # Rename to Baseline_...
        baseline_df = baseline_df.rename(
            columns={
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
                "Weeks": "Baseline_Week",
            }
        )

        # Merge back
        merged = pd.merge(df, baseline_df, on="Patient", how="left")
        return merged.drop(columns=["abs_weeks"])

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # Test df already has Baseline_ columns, but needs 'Weeks' for consistency
    if "Predict_Week" in test_df.columns:
        test_df = test_df.rename(columns={"Predict_Week": "Weeks"})

    return train_df, val_df, test_df


def get_dataloaders(debug=False):
    """
    Factory function to create dataloaders.
    """
    seed_everything(Config.SEED)

    train_df, val_df, test_df = prepare_dataframes()

    if debug:
        train_df = train_df.head(32)
        val_df = val_df.head(32)
        test_df = test_df.head(32)

    # Initialize and fit preprocessor on Training data
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    # Datasets
    train_ds = OSICDataset(train_df, preprocessor, mode="train", load_cache=True)
    val_ds = OSICDataset(val_df, preprocessor, mode="val", load_cache=True)
    test_ds = OSICDataset(test_df, preprocessor, mode="test", load_cache=True)

    # Loaders
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
