import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ProcessPoolExecutor
from library.config import Config

# Attempt to import pydicom for DICOM handling
try:
    import pydicom
except ImportError:
    pydicom = None
    print(
        "Warning: pydicom not found. DICOM processing will fail if cache is not present."
    )


class TriSlabProcessor:
    """
    Handles the conversion of 3D DICOM volumes into 2D Tri-Slab representations.
    """

    @staticmethod
    def get_lung_window(img, w=1500, l=-600):
        """Applies lung windowing to HU units."""
        lower = l - w // 2
        upper = l + w // 2
        img = np.clip(img, lower, upper)
        # Normalize to 0-1 then 0-255
        img = (img - lower) / (upper - lower)
        img = (img * 255.0).astype(np.uint8)
        return img

    @staticmethod
    def load_scan(path):
        """Loads all DICOM files from a directory and sorts them by slice location."""
        if pydicom is None:
            raise ImportError("pydicom is required to process DICOM files.")

        slices = []
        for s in os.listdir(path):
            if s.endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(path, s))
                    # Check if ImagePositionPatient exists
                    if hasattr(ds, "ImagePositionPatient"):
                        slices.append(ds)
                except Exception:
                    continue

        if not slices:
            return None

        # Sort by Z position
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Extract pixel data and stack
        try:
            # Get slope and intercept
            slope = slices[0].RescaleSlope if hasattr(slices[0], "RescaleSlope") else 1
            intercept = (
                slices[0].RescaleIntercept
                if hasattr(slices[0], "RescaleIntercept")
                else 0
            )

            image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

            # Convert to HU
            image = image * slope + intercept
            return image
        except Exception as e:
            print(f"Error processing scan at {path}: {e}")
            return None

    @staticmethod
    def generate_tri_slab(volume, axis=0):
        """
        Generates a 3-channel image using overlapping MIP slabs along the specified axis.

        Args:
            volume: 3D numpy array (D, H, W)
            axis: 0 for Axial (Depth), 1 for Coronal (Height/Y)

        Returns:
            2D numpy array (H, W, 3) or (D, W, 3) resized to (IMG_SIZE, IMG_SIZE)
        """
        # If Coronal, we need to permute so the slicing axis is 0
        if axis == 1:
            # Volume is (D, H, W). Coronal view looks at H.
            # Permute to (H, D, W)
            volume = np.transpose(volume, (1, 0, 2))

        depth = volume.shape[0]
        if depth < 3:
            # Handle edge case with very few slices by repeating
            slab1 = np.max(volume, axis=0)
            slab2 = slab1
            slab3 = slab1
        else:
            # Define slab boundaries with overlap
            # Base splits: 0.33, 0.66
            # Overlap extension: 5% of total depth (approx 15% of slab)
            margin = int(depth * 0.05)
            p1 = int(depth / 3)
            p2 = int(2 * depth / 3)

            # Slab 1: 0 to 1/3 + margin
            s1_end = min(depth, p1 + margin)
            slab1 = np.max(volume[0:s1_end], axis=0)

            # Slab 2: 1/3 - margin to 2/3 + margin
            s2_start = max(0, p1 - margin)
            s2_end = min(depth, p2 + margin)
            slab2 = np.max(volume[s2_start:s2_end], axis=0)

            # Slab 3: 2/3 - margin to end
            s3_start = max(0, p2 - margin)
            slab3 = np.max(volume[s3_start:], axis=0)

        # Apply Lung Windowing
        slab1 = TriSlabProcessor.get_lung_window(slab1)
        slab2 = TriSlabProcessor.get_lung_window(slab2)
        slab3 = TriSlabProcessor.get_lung_window(slab3)

        # Stack to RGB
        img = np.stack([slab1, slab2, slab3], axis=-1)

        # Resize
        img = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )

        return img

    @staticmethod
    def process_and_save(patient_id, dicom_dir, output_dir):
        """Worker function to process a single patient."""
        try:
            axial_path = os.path.join(output_dir, f"{patient_id}_axial.npy")
            coronal_path = os.path.join(output_dir, f"{patient_id}_coronal.npy")

            # Skip if already exists
            if os.path.exists(axial_path) and os.path.exists(coronal_path):
                return True

            # Load Volume
            full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
            volume = TriSlabProcessor.load_scan(full_path)

            if volume is None:
                # Create black images if loading fails
                black_img = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
                )
                np.save(axial_path, black_img)
                np.save(coronal_path, black_img)
                return False

            # Generate Axial Tri-Slab
            axial_img = TriSlabProcessor.generate_tri_slab(volume, axis=0)
            np.save(axial_path, axial_img)

            # Generate Coronal Tri-Slab
            coronal_img = TriSlabProcessor.generate_tri_slab(volume, axis=1)
            np.save(coronal_path, coronal_img)

            return True
        except Exception as e:
            print(f"Failed to process {patient_id}: {e}")
            return False


class LungDataset(Dataset):
    def __init__(
        self,
        df,
        cache_dir=Config.CACHE_DIR,
        transform=None,
        mode="train",
        load_cached_data=True,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            cache_dir (str): Directory to store/load processed images.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use existing cache or force reprocess.
        """
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Prepare Data (Baseline Alignment)
        self._prepare_tabular_data()

        # Cache Processing
        self._ensure_cache(load_cached_data)

    def _prepare_tabular_data(self):
        """Aligns data so that every row uses Baseline clinical features."""
        if self.mode in ["train", "val"]:
            # For training, we need to find the baseline visit for each patient
            # and broadcast those features to all visits.

            # Identify baseline rows (min Weeks per patient)
            # We assume the dataframe contains history.
            # We sort by Patient and Weeks to easily find baseline.
            self.df = self.df.sort_values(["Patient", "Weeks"])

            # Group by Patient and take the first entry as baseline
            baseline_df = self.df.groupby("Patient").first().reset_index()

            # Select relevant baseline columns
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

            # Rename to Baseline_...
            rename_map = {
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
                "Weeks": "Baseline_Week",
            }
            baseline_df = baseline_df.rename(columns=rename_map)

            # Merge back to original dataframe
            # We drop the original clinical columns to avoid confusion, keeping only target FVC and current Weeks
            self.df = pd.merge(self.df, baseline_df, on="Patient", how="left")

        elif self.mode == "test":
            # Test CSV already has Baseline_... columns from metadata generation
            pass

        # Encode Categorical Features
        # Sex: Male=0, Female=1
        self.df["Sex_Enc"] = self.df["Baseline_Sex"].apply(
            lambda x: 1 if x == "Female" else 0
        )

        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        self.df["Smoke_Enc"] = (
            self.df["Baseline_SmokingStatus"].map(smoke_map).fillna(0).astype(int)
        )

        # One-hot encoding for Smoking (3 columns)
        self.df["Smoke_0"] = (self.df["Smoke_Enc"] == 0).astype(float)
        self.df["Smoke_1"] = (self.df["Smoke_Enc"] == 1).astype(float)
        self.df["Smoke_2"] = (self.df["Smoke_Enc"] == 2).astype(float)

        # Normalize Numerical Features
        # Age: Scale roughly to 0-1 range (e.g., divide by 100)
        self.df["Age_Norm"] = self.df["Baseline_Age"] / 100.0

        # Percent: Scale roughly to 0-1 range (e.g., divide by 100)
        self.df["Percent_Norm"] = self.df["Baseline_Percent"] / 100.0

    def _ensure_cache(self, load_cached_data):
        """Checks if data is cached; if not, triggers processing."""
        unique_patients = self.df[["Patient", "dicom_dir"]].drop_duplicates()
        patients_to_process = []

        for _, row in unique_patients.iterrows():
            pid = row["Patient"]
            ax_path = os.path.join(self.cache_dir, f"{pid}_axial.npy")
            cor_path = os.path.join(self.cache_dir, f"{pid}_coronal.npy")

            if not load_cached_data or not (
                os.path.exists(ax_path) and os.path.exists(cor_path)
            ):
                patients_to_process.append((pid, row["dicom_dir"]))

        if patients_to_process:
            print(
                f"Processing {len(patients_to_process)} patients for {self.mode} set..."
            )
            # Use parallel processing
            # Note: We limit max_workers to avoid memory issues if volumes are large
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = []
                for pid, d_dir in patients_to_process:
                    futures.append(
                        executor.submit(
                            TriSlabProcessor.process_and_save,
                            pid,
                            d_dir,
                            self.cache_dir,
                        )
                    )

                # Wait for completion
                for f in futures:
                    f.result()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # Load Images
        ax_path = os.path.join(self.cache_dir, f"{pid}_axial.npy")
        cor_path = os.path.join(self.cache_dir, f"{pid}_coronal.npy")

        try:
            img_ax = np.load(ax_path)
            img_cor = np.load(cor_path)
        except FileNotFoundError:
            # Fallback for debugging or missing files
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Apply Augmentations
        if self.transform:
            # Apply same spatial transform to both views?
            # Usually independent is fine, but if we want to preserve some correlation,
            # we might use 'additional_targets'.
            # However, Axial and Coronal are different views, so independent augmentation is acceptable
            # provided it's just spatial noise.
            # Let's apply independent transforms as they are separate streams.
            aug_ax = self.transform(image=img_ax)["image"]
            aug_cor = self.transform(image=img_cor)["image"]
        else:
            # Just to tensor
            t = ToTensorV2()
            aug_ax = t(image=img_ax)["image"]
            aug_cor = t(image=img_cor)["image"]

        # Normalize Images to 0-1 float
        img_ax = aug_ax.float() / 255.0
        img_cor = aug_cor.float() / 255.0

        # Prepare Tabular Data
        # [Age, Sex, Smoke_0, Smoke_1, Smoke_2, Percent]
        tabular = torch.tensor(
            [
                row["Age_Norm"],
                row["Sex_Enc"],
                row["Smoke_0"],
                row["Smoke_1"],
                row["Smoke_2"],
                row["Percent_Norm"],
            ],
            dtype=torch.float32,
        )

        # Prepare Meta/Target
        meta = {
            "Patient": pid,
            "Baseline_FVC": float(row["Baseline_FVC"]),
            "Baseline_Week": float(row["Baseline_Week"]),
        }

        if self.mode == "test":
            # For test, we need the prediction week
            predict_week = row["Predict_Week"]
            meta["Week_Num"] = float(predict_week)
            # Target is dummy
            target = 0.0
        else:
            # For train/val, we have the current week and FVC
            current_week = row["Weeks"]
            meta["Week_Num"] = float(current_week)
            target = float(row["FVC"])

        return {
            "image_axial": img_ax,
            "image_coronal": img_cor,
            "tabular": tabular,
            "target": torch.tensor(target, dtype=torch.float32),
            "meta": meta,
        }


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Strictly spatial only - no intensity changes.
    """
    if mode == "train" and Config.USE_AUGMENTATION:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
