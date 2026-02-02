import os
import cv2
import torch
import numpy as np
import pandas as pd
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import log_message


class LungDataset(Dataset):
    """
    Dataset class for Lung Function Decline prediction.
    Handles loading of metadata, DICOM image processing (with caching),
    and generation of multimodal inputs (Axial/Coronal Tri-Slabs + Tabular).
    """

    def __init__(
        self, csv_path, mode="train", transform=None, cache_dir=Config.CACHE_DIR
    ):
        self.mode = mode
        self.csv_path = csv_path
        self.transform = transform
        self.cache_dir = cache_dir

        # Load Metadata
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # Fallback for debugging if files aren't generated yet
            print(f"Warning: {csv_path} not found.")
            self.df = pd.DataFrame()

        # Preprocess Metadata
        self._prepare_metadata()

        # Define Tabular Encoders
        self.sex_map = {"Male": 0.0, "Female": 1.0}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def _prepare_metadata(self):
        """
        Standardizes column names and prepares baseline data.
        """
        if self.mode in ["train", "val"]:
            # For training, we need to find the baseline FVC for each patient.
            # We assume the baseline is the measurement closest to Week 0.
            self.df["Abs_Week"] = self.df["Weeks"].abs()

            # Find baseline rows (min absolute week per patient)
            # We sort by patient and abs_week to easily pick the first one
            sorted_df = self.df.sort_values(["Patient", "Abs_Week"])
            baseline_df = sorted_df.groupby("Patient").first().reset_index()

            # Create a map for Baseline FVC and Week
            self.patient_baselines = dict(
                zip(baseline_df["Patient"], baseline_df["FVC"])
            )
            self.patient_base_weeks = dict(
                zip(baseline_df["Patient"], baseline_df["Weeks"])
            )

            # Drop the helper column
            self.df = self.df.drop(columns=["Abs_Week"])

        elif self.mode == "test":
            # Test CSV already has Baseline columns: Baseline_FVC, Baseline_Week, etc.
            # We standardize them to match the logic in __getitem__
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Get Images (Axial and Coronal Tri-Slabs)
        # The dicom_dir in metadata is relative (e.g., "train/ID...")
        # We join it with INPUT_DIR
        dicom_rel_path = row["dicom_dir"]
        full_dicom_path = os.path.join(Config.INPUT_DIR, dicom_rel_path)

        images = self._get_images(patient_id, full_dicom_path)

        # 2. Get Tabular Features & Meta
        if self.mode in ["train", "val"]:
            # Features
            age = row["Age"]
            sex = row["Sex"]
            smoke = row["SmokingStatus"]
            percent = row["Percent"]

            # Meta / Target
            current_week = row["Weeks"]
            fvc = row["FVC"]

            # Retrieve baseline info
            base_fvc = self.patient_baselines.get(patient_id, fvc)
            base_week = self.patient_base_weeks.get(patient_id, current_week)

            dt = current_week - base_week

        else:  # Test mode
            # Features (prefixed with Baseline_)
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
            percent = row["Baseline_Percent"]

            # Meta
            current_week = row["Predict_Week"]
            base_week = row["Baseline_Week"]
            base_fvc = row["Baseline_FVC"]

            dt = current_week - base_week
            fvc = 0.0  # Dummy for test

        # 3. Construct Tabular Vector
        # [Age_norm, Sex_bin, Smoke_0, Smoke_1, Smoke_2, Percent_norm]
        # Normalization constants derived from EDA
        age_norm = (age - 65.0) / 15.0
        percent_norm = percent / 100.0
        sex_enc = self.sex_map.get(sex, 0.5)

        smoke_idx = self.smoke_map.get(smoke, 0)
        smoke_ohe = [0.0, 0.0, 0.0]
        smoke_ohe[smoke_idx] = 1.0

        tabular_vec = np.array(
            [age_norm, sex_enc] + smoke_ohe + [percent_norm], dtype=np.float32
        )

        # 4. Apply Transforms to Images
        if self.transform:
            # Albumentations expects a single image or dictionary.
            # We apply same spatial transform to both views if possible,
            # but they are different views. Independent augmentation is acceptable
            # as long as it's spatial only.

            # Apply to Axial
            aug_ax = self.transform(image=images["axial"])["image"]
            # Apply to Coronal
            aug_cor = self.transform(image=images["coronal"])["image"]

            images_t = {"axial": aug_ax, "coronal": aug_cor}
        else:
            # Convert to tensor manually if no transform
            t_ax = torch.tensor(images["axial"].transpose(2, 0, 1), dtype=torch.float32)
            t_cor = torch.tensor(
                images["coronal"].transpose(2, 0, 1), dtype=torch.float32
            )
            images_t = {"axial": t_ax, "coronal": t_cor}

        return {
            "patient_id": patient_id,
            "img_axial": images_t["axial"],
            "img_coronal": images_t["coronal"],
            "tabular": torch.tensor(tabular_vec, dtype=torch.float32),
            "meta_dt": torch.tensor(dt, dtype=torch.float32),
            "meta_base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "target": torch.tensor(fvc, dtype=torch.float32),
        }

    def _get_images(self, patient_id, dicom_dir):
        """
        Retrieves processed images, using cache if available.
        """
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # Check Cache
        if os.path.exists(axial_path) and os.path.exists(coronal_path):
            try:
                ax_img = np.load(axial_path)
                cor_img = np.load(coronal_path)
                return {"axial": ax_img, "coronal": cor_img}
            except Exception as e:
                print(f"Error loading cache for {patient_id}: {e}. Reprocessing.")

        # Process from Scratch
        try:
            volume = self._read_dicom_volume(dicom_dir)
            if volume is None:
                # Fallback for empty/corrupt directories: Black image
                ax_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
                cor_img = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
                )
            else:
                ax_img = self._generate_tri_slab(volume, axis=0)  # Z-axis (Axial)
                cor_img = self._generate_tri_slab(volume, axis=1)  # Y-axis (Coronal)

            # Save to Cache
            np.save(axial_path, ax_img)
            np.save(coronal_path, cor_img)

            return {"axial": ax_img, "coronal": cor_img}

        except Exception as e:
            print(f"Error processing DICOM for {patient_id}: {e}")
            # Return zeros on failure
            ax_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            cor_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            return {"axial": ax_img, "coronal": cor_img}

    def _read_dicom_volume(self, path):
        """
        Reads a directory of DICOM files and constructs a 3D volume.
        Returns: numpy array of shape (Z, Y, X)
        """
        if not os.path.exists(path):
            return None

        files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        if not files:
            return None

        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, f))
                # Ensure we have image position for sorting
                if hasattr(ds, "ImagePositionPatient"):
                    slices.append(ds)
            except Exception:
                continue

        if not slices:
            return None

        # Sort by Z position (ImagePositionPatient[2])
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Stack images
        # Handle RescaleSlope and Intercept if present to get Hounsfield Units
        images = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = slope * img + intercept
            images.append(img)

        volume = np.stack(images)  # (Z, H, W)
        return volume

    def _generate_tri_slab(self, volume, axis):
        """
        Generates Fixed Overlapping Tri-Slabs (MIPs) along a specific axis.
        Args:
            volume: 3D numpy array (Z, Y, X)
            axis: 0 for Axial (Z), 1 for Coronal (Y)
        Returns:
            224x224x3 uint8 image
        """
        # If Coronal, we need to handle the shape.
        # Volume is (Z, Y, X).
        # If axis=0 (Axial), we slice along Z. Result (Y, X).
        # If axis=1 (Coronal), we slice along Y. Result (Z, X).

        depth = volume.shape[axis]

        # Define Slab Indices with 15% overlap
        # We divide depth into 3 parts.
        # Part size approx D/3.
        # Overlap pixels = D * 0.15

        p1 = int(depth / 3)
        p2 = int(2 * depth / 3)
        overlap = int(depth * Config.SLAB_OVERLAP)

        # Define 3 intervals
        # Slab 1: 0 -> p1 + overlap
        s1_start, s1_end = 0, min(depth, p1 + overlap)

        # Slab 2: p1 - overlap -> p2 + overlap
        s2_start, s2_end = max(0, p1 - overlap), min(depth, p2 + overlap)

        # Slab 3: p2 - overlap -> end
        s3_start, s3_end = max(0, p2 - overlap), depth

        # Helper to slice and MIP
        def get_mip(start, end):
            if start >= end:
                # Fallback for very thin volumes
                start = 0
                end = depth

            # Slice along the specified axis
            if axis == 0:
                slab = volume[start:end, :, :]
                mip = np.max(slab, axis=0)  # (Y, X)
            else:
                slab = volume[:, start:end, :]
                mip = np.max(slab, axis=1)  # (Z, X)

                # For Coronal (Z, X), Z is usually smaller than X and Y.
                # We might need to resize to match aspect ratio or just resize to target.
                # Standard Coronal view: Z is vertical, X is horizontal.
                # But in numpy (Z, Y, X), Z is axis 0 (vertical in array), X is axis 2.
                # So (Z, X) is correct orientation for image processing.

            return mip

        ch1 = get_mip(s1_start, s1_end)
        ch2 = get_mip(s2_start, s2_end)
        ch3 = get_mip(s3_start, s3_end)

        # Stack to RGB (H, W, 3)
        img = np.stack([ch1, ch2, ch3], axis=-1)

        # Normalize to 0-255 uint8
        # Lung window is usually -1000 to 400 approx, but here we use min/max of volume
        # Robust min-max normalization
        v_min, v_max = -1000, 400  # Standard Lung Window
        img = np.clip(img, v_min, v_max)
        img = (img - v_min) / (v_max - v_min)
        img = (img * 255).astype(np.uint8)

        # Resize
        img = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
        )

        return img


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    Strictly Spatial Only - No intensity augmentations.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def process_dicom(
    patient_id, dicom_dir, cache_dir=Config.CACHE_DIR, load_cached_data=True
):
    """
    Wrapper function to process DICOMs for a patient.
    Used for debugging or manual processing.
    """
    dataset = LungDataset(Config.TRAIN_CSV, mode="train", cache_dir=cache_dir)
    return dataset._get_images(patient_id, dicom_dir)
