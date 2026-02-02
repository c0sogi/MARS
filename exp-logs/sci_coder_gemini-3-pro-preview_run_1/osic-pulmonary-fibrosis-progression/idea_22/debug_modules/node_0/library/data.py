import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_instance_number(s):
    """Safely extract InstanceNumber from DICOM header."""
    try:
        return int(s.InstanceNumber)
    except:
        return 0


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by InstanceNumber.
    """
    if not os.path.exists(path):
        return []

    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return []

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except:
            continue

    slices.sort(key=lambda x: get_instance_number(x))
    return slices


def get_pixels_hu(slices):
    """
    Converts DICOM slices to a 3D numpy array of Hounsfield Units.
    """
    if not slices:
        return np.zeros((1, 224, 224), dtype=np.int16)

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU using RescaleSlope and RescaleIntercept
    for i, s in enumerate(slices):
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        image[i] = slope * image[i].astype(np.float64) + intercept

    return image


def normalize_hu(image):
    """
    Normalize Hounsfield Units to 0-255 range using Lung Window.
    Window: Width=1500, Level=-600 -> Range: [-1350, 150]
    """
    min_hu = -1350
    max_hu = 150

    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    image = (image * 255).astype(np.uint8)
    return image


def generate_tri_slab(volume, axis=0, size=224):
    """
    Generates a 3-channel image using overlapping MIP slabs.

    Args:
        volume: 3D numpy array (Z, Y, X)
        axis: 0 for Axial (Z-axis), 1 for Coronal (Y-axis)
        size: Target spatial resolution

    Returns:
        (size, size, 3) numpy array, normalized to 0-255 uint8
    """
    if volume.ndim != 3 or volume.shape[0] == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Permute volume so the target axis is at dim 0 (Depth)
    # Axial (Z): already (Z, Y, X)
    # Coronal (Y): (Z, Y, X) -> (Y, Z, X)
    if axis == 1:
        vol_perm = np.transpose(volume, (1, 0, 2))
    else:
        vol_perm = volume

    depth = vol_perm.shape[0]

    # Handle cases with very few slices
    if depth < 3:
        mip = np.max(vol_perm, axis=0)
        mip = normalize_hu(mip)
        img = cv2.resize(mip, (size, size))
        return np.stack([img, img, img], axis=-1)

    # Define slab boundaries with 15% overlap
    p1 = depth / 3.0
    p2 = 2.0 * depth / 3.0
    overlap = depth * 0.15
    half_overlap = overlap / 2.0

    # Calculate indices
    start1, end1 = 0, int(p1 + half_overlap)
    start2, end2 = int(p1 - half_overlap), int(p2 + half_overlap)
    start3, end3 = int(p2 - half_overlap), depth

    # Clamp indices
    start1, end1 = max(0, start1), min(depth, end1)
    start2, end2 = max(0, start2), min(depth, end2)
    start3, end3 = max(0, start3), min(depth, end3)

    # Compute MIPs for each slab
    slab1 = np.max(vol_perm[start1:end1, :, :], axis=0)
    slab2 = np.max(vol_perm[start2:end2, :, :], axis=0)
    slab3 = np.max(vol_perm[start3:end3, :, :], axis=0)

    # Normalize and stack
    slab1 = normalize_hu(slab1)
    slab2 = normalize_hu(slab2)
    slab3 = normalize_hu(slab3)

    img = np.stack([slab1, slab2, slab3], axis=-1)

    # Resize to target resolution
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    return img


class LungDataset(Dataset):
    def __init__(self, df, images, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.images = images  # Shape (N, 2, H, W, 3)
        self.transforms = transforms
        self.mode = mode
        self.tabular_feats = self._process_tabular()

    def _process_tabular(self):
        # Handle feature extraction for both Train and Test schemas
        if "Age" in self.df.columns:
            age = self.df["Age"].values
            pct = self.df["Percent"].values
            smk = self.df["SmokingStatus"].values
            # Encode Sex
            sex_map = {"Male": 0, "Female": 1}
            sex = self.df["Sex"].map(sex_map).fillna(0).values
        else:
            age = self.df["Baseline_Age"].values
            pct = self.df["Baseline_Percent"].values
            smk = self.df["Baseline_SmokingStatus"].values
            sex_map = {"Male": 0, "Female": 1}
            sex = self.df["Baseline_Sex"].map(sex_map).fillna(0).values

        # Normalize Age: (x - 50) / 50
        age_norm = (age - 50.0) / 50.0

        # Normalize Percent: (x - 50) / 100
        pct_norm = (pct - 50.0) / 100.0

        # One-hot encode SmokingStatus
        smk_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        smk_idx = np.array([smk_map.get(s, 0) for s in smk])
        smk_oh = np.eye(3)[smk_idx]

        # Normalize Baseline FVC (Critical for Prior-Anchored Head)
        # We assume Baseline_FVC is present in df (added during preparation)
        base_fvc = self.df["Baseline_FVC"].values
        base_fvc_norm = (base_fvc - 2500.0) / 1000.0

        # Concatenate: Age(1) + Sex(1) + Smoking(3) + Percent(1) + BaseFVC(1) = 7 features
        feats = np.column_stack(
            [age_norm, sex, smk_oh, pct_norm, base_fvc_norm]
        ).astype(np.float32)
        return feats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve images: 0=Axial, 1=Coronal
        imgs = self.images[idx]
        ax_img = imgs[0]
        cor_img = imgs[1]

        # Apply transforms
        if self.transforms:
            # Apply independent transforms to learn robust features
            aug_ax = self.transforms(image=ax_img)["image"]
            aug_cor = self.transforms(image=cor_img)["image"]
        else:
            t = ToTensorV2()
            aug_ax = t(image=ax_img)["image"]
            aug_cor = t(image=cor_img)["image"]

        tab = torch.tensor(self.tabular_feats[idx], dtype=torch.float32)
        row = self.df.iloc[idx]

        if self.mode == "test":
            meta = {
                "Baseline_FVC": float(row["Baseline_FVC"]),
                "Baseline_Week": int(row["Baseline_Week"]),
                "Predict_Week": int(row["Predict_Week"]),
                "Patient_Week": row["Patient_Week"],
            }
            return {"axial": aug_ax, "coronal": aug_cor, "tabular": tab, "meta": meta}
        else:
            target = torch.tensor(row["FVC"], dtype=torch.float32)
            week = torch.tensor(row["Weeks"], dtype=torch.float32)
            return {
                "axial": aug_ax,
                "coronal": aug_cor,
                "tabular": tab,
                "target": target,
                "week": week,
                "patient": row["Patient"],
            }


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
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


def prepare_data(subset="train", load_cached_data=True):
    """
    Loads metadata, processes images (or loads cache), returns Dataset.
    """
    # Define paths based on subset
    if subset == "train":
        csv_path = Config.TRAIN_CSV
        cache_img_path = os.path.join(Config.CACHE_DIR, "train_images.npy")
        cache_df_path = os.path.join(Config.CACHE_DIR, "train_meta.parquet")
    elif subset == "val":
        csv_path = Config.VAL_CSV
        cache_img_path = os.path.join(Config.CACHE_DIR, "val_images.npy")
        cache_df_path = os.path.join(Config.CACHE_DIR, "val_meta.parquet")
    else:
        csv_path = Config.TEST_CSV
        cache_img_path = os.path.join(Config.CACHE_DIR, "test_images.npy")
        cache_df_path = os.path.join(Config.CACHE_DIR, "test_meta.parquet")

    # Attempt to load cache
    if (
        load_cached_data
        and os.path.exists(cache_img_path)
        and os.path.exists(cache_df_path)
    ):
        print(f"Loading cached {subset} data from {Config.CACHE_DIR}...")
        try:
            images = np.load(cache_img_path)
            df = pd.read_parquet(cache_df_path)
            return LungDataset(
                df, images, transforms=get_transforms(subset), mode=subset
            )
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # Load raw metadata
    df = pd.read_csv(csv_path)

    # Ensure Baseline_FVC exists for train/val
    if subset != "test":
        # We approximate baseline FVC using the first recorded FVC (min Weeks) for each patient
        # This is needed for the Prior-Anchored Head
        baseline_map = df.sort_values("Weeks").groupby("Patient")["FVC"].first()
        df["Baseline_FVC"] = df["Patient"].map(baseline_map)
        # Also need Baseline_Week for consistency in logic if needed, though usually 0
        baseline_week_map = df.sort_values("Weeks").groupby("Patient")["Weeks"].first()
        df["Baseline_Week"] = df["Patient"].map(baseline_week_map)

    print(f"Processing {subset} images (DICOM to Tri-Slab)...")

    image_list = []
    patient_cache = (
        {}
    )  # Cache images by PatientID to avoid re-reading DICOMs for multiple rows

    for idx, row in df.iterrows():
        pid = row["Patient"]

        if pid in patient_cache:
            image_list.append(patient_cache[pid])
            continue

        # Construct path to DICOM directory
        # metadata 'dicom_dir' is relative to INPUT_DIR
        dcm_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])

        # Load Volume
        slices = load_scan(dcm_dir)
        vol = get_pixels_hu(slices)  # (Z, Y, X)

        # Generate Orthogonal Views
        ax_view = generate_tri_slab(vol, axis=0, size=Config.IMG_SIZE)  # (224, 224, 3)
        cor_view = generate_tri_slab(vol, axis=1, size=Config.IMG_SIZE)  # (224, 224, 3)

        # Stack views: (2, 224, 224, 3)
        imgs = np.stack([ax_view, cor_view], axis=0)

        patient_cache[pid] = imgs
        image_list.append(imgs)

    images_arr = np.array(image_list, dtype=np.uint8)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_img_path, images_arr)
    df.to_parquet(cache_df_path)
    print(f"Cached {subset} data to {Config.CACHE_DIR}")

    return LungDataset(df, images_arr, transforms=get_transforms(subset), mode=subset)
