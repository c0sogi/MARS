import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def get_img_transforms(mode="train"):
    """
    Returns albumentations transforms.
    Train: Spatial augmentations + Normalization.
    Val/Test: Normalization only.
    """
    if mode == "train" and Config.USE_AUGMENTATION:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
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


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by slice location.
    """
    slices = []
    try:
        for s in os.listdir(path):
            if s.endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(path, s))
                    # Ensure necessary attributes exist
                    if hasattr(ds, "ImagePositionPatient"):
                        slices.append(ds)
                except Exception:
                    continue

        if not slices:
            return None

        # Sort by ImagePositionPatient Z coordinate
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Check for slice thickness and infer if missing
        try:
            slice_thickness = np.abs(
                slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
            )
        except:
            slice_thickness = getattr(slices[0], "SliceThickness", 1.0)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices
    except Exception as e:
        print(f"Error loading scan from {path}: {e}")
        return None


def get_pixels_hu(slices):
    """
    Converts DICOM raw data to Hounsfield Units.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    intercept = (
        slices[0].RescaleIntercept if hasattr(slices[0], "RescaleIntercept") else -1024
    )
    slope = slices[0].RescaleSlope if hasattr(slices[0], "RescaleSlope") else 1

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    # Handle padding values (often -2000)
    image[image < -1000] = -1000
    return image


def apply_lung_window(image, center=-600, width=1500):
    """
    Applies lung windowing to HU image and normalizes to 0-255.
    """
    min_value = center - width // 2
    max_value = center + width // 2
    image = np.clip(image, min_value, max_value)
    image = (image - min_value) / (max_value - min_value)
    return image


def generate_trislab_image(volume, view="axial"):
    """
    Generates a 3-channel image from a 3D volume using overlapping slabs and MIP.

    Args:
        volume: 3D numpy array (Z, Y, X)
        view: 'axial' or 'coronal'
    Returns:
        2D numpy array (H, W, 3) normalized to 0-1
    """
    # Reslice if Coronal
    if view == "coronal":
        # Transpose to make Y the depth axis: (Z, Y, X) -> (Y, Z, X)
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]
    num_slabs = Config.NUM_SLABS
    overlap = Config.SLAB_OVERLAP

    # Calculate slab parameters
    # Total coverage needed is depth.
    # Let slab_size be S. Effective coverage = S + (S-overlap*S) + ...
    # Simple approach: Divide depth into N equal parts with overlap padding

    if depth < num_slabs:
        # Edge case: very few slices. Duplicate.
        slabs = [np.max(volume, axis=0)] * num_slabs
    else:
        # Calculate slab size to cover the volume
        # We want 3 slabs.
        # Slab 1: 0 to end1
        # Slab 2: start2 to end2
        # Slab 3: start3 to depth
        # With overlap.

        # Basic strategy: Divide index space into 3 chunks
        chunk_size = depth / num_slabs
        overlap_size = int(chunk_size * overlap)

        slabs = []
        for i in range(num_slabs):
            start = max(0, int(i * chunk_size) - overlap_size)
            end = min(depth, int((i + 1) * chunk_size) + overlap_size)

            # Extract slab
            slab_vol = volume[start:end, :, :]

            if slab_vol.shape[0] == 0:
                # Fallback
                mip = np.zeros((volume.shape[1], volume.shape[2]))
            else:
                # Compute MIP
                mip = np.max(slab_vol, axis=0)
            slabs.append(mip)

    # Stack into channels
    img = np.stack(slabs, axis=-1)  # (H, W, 3)

    # Resize to target size
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    return img


def process_dicom_trislab(
    patient_id, dicom_dir, cache_dir=Config.CACHE_DIR, load_cached_data=True
):
    """
    End-to-end processing for a patient's CT scan.
    Returns Axial and Coronal Tri-Slab images.
    """
    # Define cache paths
    cache_path_ax = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    cache_path_cor = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Load Cache
    if (
        load_cached_data
        and os.path.exists(cache_path_ax)
        and os.path.exists(cache_path_cor)
    ):
        try:
            img_ax = np.load(cache_path_ax)
            img_cor = np.load(cache_path_cor)
            return img_ax, img_cor
        except Exception:
            pass  # Failed to load, recompute

    # 2. Compute from Scratch
    full_path = os.path.join(Config.INPUT_DIR, dicom_dir)

    # Handle case where directory doesn't exist (e.g. bad path in csv)
    if not os.path.exists(full_path):
        # Return black images
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return img, img

    slices = load_scan(full_path)
    if not slices:
        # Return black images
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return img, img

    try:
        vol_hu = get_pixels_hu(slices)
        vol_norm = apply_lung_window(vol_hu)  # 0-1 float

        # Generate Views
        img_ax = generate_trislab_image(vol_norm, view="axial")
        img_cor = generate_trislab_image(vol_norm, view="coronal")

        # Ensure float32
        img_ax = img_ax.astype(np.float32)
        img_cor = img_cor.astype(np.float32)
    except (RuntimeError, Exception) as e:
        print(
            f"Warning: Error processing {patient_id} (likely decompression). Using black image fallback. Error: {e}"
        )
        img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # 3. Save Cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path_ax, img_ax)
        np.save(cache_path_cor, img_cor)

    return img_ax, img_cor


# ==========================================
# Dataset Class
# ==========================================


class LungDataset(Dataset):
    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode: 'train', 'val', or 'test'
            transform: albumentations transforms
        """
        self.mode = mode
        self.transform = transform

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)

        # Debugging
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Subsampling {mode} dataset to {Config.DEBUG_DATA_LIMIT} rows."
            )
            self.df = self.df.head(Config.DEBUG_DATA_LIMIT)

        # Preprocess Tabular Data
        self._prepare_tabular_data()

    def _prepare_tabular_data(self):
        """
        Encodes categorical features and normalizes numerical ones.
        Also handles baseline feature extraction for training.
        """
        # Categorical Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        if self.mode in ["train", "val"]:
            # For training, we need to identify the baseline characteristics for each patient.
            # We assume the visit with min(|Weeks|) is the baseline.

            # Create a baseline lookup
            self.df["abs_weeks"] = self.df["Weeks"].abs()
            baseline_df = (
                self.df.sort_values("abs_weeks")
                .groupby("Patient")
                .first()
                .reset_index()
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

            # Merge baseline info back to the main dataframe
            # We drop existing columns if they conflict or just use the suffixed ones
            self.df = pd.merge(self.df, baseline_df, on="Patient", how="left")

        elif self.mode == "test":
            # Test CSV already has Baseline_FVC, Baseline_Percent, etc.
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial + Coronal)
        img_ax, img_cor = process_dicom_trislab(
            patient_id, row["dicom_dir"], load_cached_data=True
        )

        # 2. Apply Augmentations
        if self.transform:
            # Apply same spatial transform to both?
            # Usually independent is fine for classification, but here they are views of same person.
            # However, they are different coordinate systems. Independent is safer.
            res_ax = self.transform(image=img_ax)
            img_ax_t = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor_t = res_cor["image"]
        else:
            # Fallback to simple tensor conversion
            t = ToTensorV2()
            img_ax_t = t(image=img_ax)["image"]
            img_cor_t = t(image=img_cor)["image"]

        # 3. Prepare Tabular Vector
        # Features: [Age, Sex, Smoking, Percent]
        # We use Baseline values for the model input

        age = float(row["Baseline_Age"])
        sex = self.sex_map.get(row["Baseline_Sex"], 0)
        smoke = self.smoke_map.get(row["Baseline_SmokingStatus"], 1)
        percent = float(row["Baseline_Percent"])

        # Simple normalization for stability
        # Age ~ 0-100 -> /100
        # Percent ~ 0-150 -> /100
        # Sex, Smoke are categorical indices? Or one-hot?
        # The idea says "Raw Tabular Features... projected to 1280-dim".
        # We will provide a vector and let the model handle embedding/projection.

        tab_vec = np.array(
            [age / 100.0, float(sex), float(smoke), percent / 100.0], dtype=np.float32
        )

        # 4. Prepare Targets / Metadata
        if self.mode in ["train", "val"]:
            target_fvc = float(row["FVC"])
            current_week = int(row["Weeks"])
            baseline_week = int(row["Baseline_Week"])
            baseline_fvc = float(row["Baseline_FVC"])

            week_delta = current_week - baseline_week

            return {
                "img_ax": img_ax_t,
                "img_cor": img_cor_t,
                "tabular": torch.tensor(tab_vec, dtype=torch.float32),
                "week_delta": torch.tensor(week_delta, dtype=torch.float32),
                "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
                "target_fvc": torch.tensor(target_fvc, dtype=torch.float32),
            }

        else:  # Test
            # For test, we need to predict for 'Predict_Week'
            predict_week = int(row["Predict_Week"])
            baseline_week = int(row["Baseline_Week"])
            baseline_fvc = float(row["Baseline_FVC"])

            week_delta = predict_week - baseline_week

            return {
                "img_ax": img_ax_t,
                "img_cor": img_cor_t,
                "tabular": torch.tensor(tab_vec, dtype=torch.float32),
                "week_delta": torch.tensor(week_delta, dtype=torch.float32),
                "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
                "patient_week": row["Patient_Week"],
            }


# ==========================================
# Data Loaders
# ==========================================


def get_dataloaders():
    """
    Creates and returns training and validation dataloaders.
    """
    train_transform = get_img_transforms(mode="train")
    val_transform = get_img_transforms(mode="val")

    train_dataset = LungDataset(mode="train", transform=train_transform)
    val_dataset = LungDataset(mode="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates and returns the test dataloader.
    """
    test_transform = get_img_transforms(mode="test")
    test_dataset = LungDataset(mode="test", transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
