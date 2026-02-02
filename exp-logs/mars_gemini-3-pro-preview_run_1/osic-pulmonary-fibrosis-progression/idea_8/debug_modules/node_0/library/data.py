import os
import glob
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Attempt to import pydicom; essential for this task
try:
    import pydicom
except ImportError:
    pydicom = None
    print("Warning: pydicom not found. DICOM processing may fail.")


class LungDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing patient metadata.
            cache_dir (str): Directory to save/load processed images.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

        # Feature Encoders
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial and Coronal Tri-Slabs)
        # We rely on the caching mechanism to provide these
        images = self._load_patient_images(patient_id, row["dicom_dir"])

        img_ax = images["axial"]
        img_cor = images["coronal"]

        # 2. Apply Augmentations (Spatial Only)
        if self.transform:
            # Apply same geometric transform to both views if possible,
            # but they are different views, so independent augmentation is acceptable
            # and often better for regularization.
            aug_ax = self.transform(image=img_ax)["image"]
            aug_cor = self.transform(image=img_cor)["image"]
        else:
            # Just convert to tensor
            base_t = ToTensorV2()
            aug_ax = base_t(image=img_ax)["image"]
            aug_cor = base_t(image=img_cor)["image"]

        # 3. Prepare Tabular Features
        # Inputs: Age, Sex (OH), Smoking (OH), Baseline_Percent
        # Normalize Age (approx 0-1 range based on typical data)
        age_norm = (row["Baseline_Age"] - 50) / 50.0
        pct_norm = row["Baseline_Percent"] / 100.0

        # One-Hot Encoding
        sex_oh = np.zeros(2, dtype=np.float32)
        sex_idx = self.sex_map.get(row["Baseline_Sex"], 0)
        sex_oh[sex_idx] = 1.0

        smoke_oh = np.zeros(3, dtype=np.float32)
        smoke_idx = self.smoke_map.get(row["Baseline_SmokingStatus"], 0)
        smoke_oh[smoke_idx] = 1.0

        # Concatenate: Age(1) + Sex(2) + Smoke(3) + Pct(1) = 7
        tabular = np.concatenate(
            [
                np.array([age_norm], dtype=np.float32),
                sex_oh,
                smoke_oh,
                np.array([pct_norm], dtype=np.float32),
            ]
        )

        # 4. Prepare Targets / Context
        # For inference, we need Baseline_FVC and the relative week
        # 'Weeks' in the DF is the target week relative to baseline
        # 'Baseline_Week' is usually 0, but we use the delta.

        if self.mode == "test":
            # In test.csv, 'Weeks' column was renamed to 'Predict_Week' in metadata generation
            # We assume the DF passed here has 'Weeks' representing the target week
            target_week = row["Weeks"]
        else:
            target_week = row["Weeks"]

        baseline_week = row.get("Baseline_Week", 0)
        relative_week = target_week - baseline_week

        baseline_fvc = row["Baseline_FVC"]

        data = {
            "img_ax": aug_ax,  # (3, 224, 224)
            "img_cor": aug_cor,  # (3, 224, 224)
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "relative_week": torch.tensor(relative_week, dtype=torch.float32),
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "patient_id": patient_id,
        }

        if self.mode != "test":
            data["fvc_target"] = torch.tensor(row["FVC"], dtype=torch.float32)

        return data

    def _load_patient_images(self, patient_id, rel_dicom_dir):
        """
        Handles caching logic.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                return data
            except Exception as e:
                print(f"Error loading cache for {patient_id}: {e}. Reprocessing.")

        # Process from scratch
        full_dicom_path = os.path.join(Config.INPUT_DIR, rel_dicom_dir)
        processed_data = process_dicom_volume(full_dicom_path)

        # Save to cache
        try:
            np.save(cache_path, processed_data)
        except Exception as e:
            print(f"Failed to save cache for {patient_id}: {e}")

        return processed_data


def process_dicom_volume(path):
    """
    Reads DICOM files, generates Axial and Coronal Tri-Slabs.
    Returns:
        dict: {'axial': np.array (H, W, 3), 'coronal': np.array (H, W, 3)}
              Values are uint8 [0, 255].
    """
    if not os.path.exists(path):
        # Return blank images if path doesn't exist (robustness)
        blank = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        return {"axial": blank, "coronal": blank}

    files = sorted(glob.glob(os.path.join(path, "*.dcm")))
    if not files:
        blank = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        return {"axial": blank, "coronal": blank}

    slices = []
    for f in files:
        try:
            if pydicom:
                dcm = pydicom.dcmread(f)
                # Extract Z position for sorting
                try:
                    z_pos = float(dcm.ImagePositionPatient[2])
                except:
                    z_pos = 0

                # Convert to HU
                img = dcm.pixel_array.astype(np.float32)
                slope = getattr(dcm, "RescaleSlope", 1)
                intercept = getattr(dcm, "RescaleIntercept", -1024)
                img = img * slope + intercept

                slices.append((z_pos, img))
            else:
                # Fallback (unlikely to work for raw dcm but required for safety)
                pass
        except Exception:
            continue

    if not slices:
        blank = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        return {"axial": blank, "coronal": blank}

    # Sort by Z position
    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices])  # (D, H, W)

    # Helper for windowing and normalization
    def window_and_normalize(vol):
        # Lung Window: W=1500, L=-600
        # Range: [-1350, 150]
        lower = -600 - 1500 / 2
        upper = -600 + 1500 / 2
        vol = np.clip(vol, lower, upper)
        vol = (vol - lower) / (upper - lower)
        return vol

    volume = window_and_normalize(volume)

    # --- Generate Axial Tri-Slab ---
    # Volume is (D, H, W). Split D.
    D, H, W = volume.shape

    def get_tri_slab(vol_3d, axis_idx):
        """
        Splits 3D volume along axis_idx into 3 overlapping slabs, MIPs them.
        """
        # Move target axis to 0
        if axis_idx != 0:
            vol_3d = np.moveaxis(vol_3d, axis_idx, 0)

        depth = vol_3d.shape[0]
        if depth < 3:
            # Handle edge case with very few slices
            mip = np.max(vol_3d, axis=0)
            return np.stack([mip, mip, mip], axis=-1)

        # Calculate slab boundaries with overlap
        # We want 3 slabs covering [0, depth]
        # Base slab size
        slab_size = depth / 3.0
        overlap_size = slab_size * Config.SLAB_OVERLAP

        # Define ranges
        # Slab 1: 0 -> 1/3 + overlap
        s1_end = int(slab_size + overlap_size)
        slab1 = vol_3d[0:s1_end, :, :]

        # Slab 2: 1/3 - overlap -> 2/3 + overlap
        s2_start = int(slab_size - overlap_size)
        s2_end = int(2 * slab_size + overlap_size)
        slab2 = vol_3d[s2_start:s2_end, :, :]

        # Slab 3: 2/3 - overlap -> end
        s3_start = int(2 * slab_size - overlap_size)
        slab3 = vol_3d[s3_start:, :, :]

        # MIP
        m1 = np.max(slab1, axis=0) if slab1.size > 0 else np.zeros_like(vol_3d[0])
        m2 = np.max(slab2, axis=0) if slab2.size > 0 else np.zeros_like(vol_3d[0])
        m3 = np.max(slab3, axis=0) if slab3.size > 0 else np.zeros_like(vol_3d[0])

        # Stack to RGB (H, W, 3)
        img = np.stack([m1, m2, m3], axis=-1)
        return img

    # Axial (Axis 0 is Depth)
    img_ax = get_tri_slab(volume, 0)

    # Coronal (Axis 1 is Anterior-Posterior usually in DICOM, but depends on orientation)
    # Assuming standard (D, H, W), Coronal is usually viewing along axis 1.
    img_cor = get_tri_slab(volume, 1)

    # Resize to target size
    def resize_img(img):
        # img is (H, W, 3) float [0,1]
        img_u8 = (img * 255).astype(np.uint8)
        if img_u8.shape[0] != Config.IMG_SIZE or img_u8.shape[1] != Config.IMG_SIZE:
            img_u8 = cv2.resize(
                img_u8, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
        return img_u8

    return {"axial": resize_img(img_ax), "coronal": resize_img(img_cor)}


def get_dataloaders(config):
    """
    Creates DataLoaders for train, val, and test sets.
    Preprocesses metadata to ensure uniform 'Baseline_' columns.
    """
    seed_everything(config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Preprocess Train/Val to add Baseline info
    # We need to identify the baseline visit for each patient (min Weeks)
    def add_baseline_info(df):
        # Find baseline row for each patient
        # Sort by Weeks to ensure first is baseline
        df = df.sort_values(["Patient", "Weeks"])

        # Group by Patient and take the first row as baseline
        baseline_df = df.groupby("Patient").first().reset_index()

        # Select relevant columns and rename
        cols_to_keep = [
            "Patient",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
        ]
        baseline_df = baseline_df[cols_to_keep]
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

        # Merge back to original dataframe
        merged_df = pd.merge(df, baseline_df, on="Patient", how="left")
        return merged_df

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # Test DF already has Baseline columns, but target week is 'Predict_Week'
    # Rename for consistency
    if "Predict_Week" in test_df.columns:
        test_df = test_df.rename(columns={"Predict_Week": "Weeks"})

    # 3. Define Augmentations
    # Spatial only: Flip, ShiftScaleRotate. No brightness/contrast.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 4. Create Datasets
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    train_ds = LungDataset(
        train_df, config.CACHE_DIR, mode="train", transform=train_transform
    )
    val_ds = LungDataset(val_df, config.CACHE_DIR, mode="val", transform=val_transform)
    test_ds = LungDataset(
        test_df, config.CACHE_DIR, mode="test", transform=val_transform
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
