import os
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from library.utils import seed_everything


class LungDataProcessor:
    def __init__(self, cache_dir="./working/idea_13/"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Image Parameters
        self.IMG_SIZE = 224
        # Standard Lung Window Settings
        self.WL = -600  # Window Level
        self.WW = 1500  # Window Width

        # Tabular Preprocessors
        self.num_scaler = RobustScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False

    def load_dicom_stack(self, dicom_dir):
        """
        Reads a directory of DICOM files, sorts them by Z-position,
        and converts to a 3D numpy array in Hounsfield Units.
        """
        if not os.path.exists(dicom_dir):
            # Return dummy volume if path doesn't exist (safety)
            return np.zeros((10, 512, 512), dtype=np.float32)

        files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        if not files:
            return np.zeros((10, 512, 512), dtype=np.float32)

        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(dicom_dir, f))
                # We need ImagePositionPatient to sort slices spatially
                if hasattr(ds, "ImagePositionPatient"):
                    pos = float(ds.ImagePositionPatient[2])
                    slices.append((pos, ds))
            except Exception:
                # Skip corrupted files
                continue

        # Sort slices by Z-position
        slices.sort(key=lambda x: x[0])

        images = []
        for _, ds in slices:
            try:
                img = ds.pixel_array.astype(np.float32)

                # Convert to Hounsfield Units (HU)
                intercept = getattr(ds, "RescaleIntercept", -1024)
                slope = getattr(ds, "RescaleSlope", 1)
                img = img * slope + intercept
                images.append(img)
            except Exception:
                continue

        if not images:
            return np.zeros((10, 512, 512), dtype=np.float32)

        # Stack to (Depth, Height, Width)
        return np.stack(images)

    def generate_tri_slab(self, volume, axis_name):
        """
        Generates a 3-channel RGB image representing the volume.
        Splits the volume into 3 overlapping slabs along the specified axis
        and computes the Maximum Intensity Projection (MIP) for each.
        """
        # Determine axis dimension
        # Volume is (Z, Y, X)
        if axis_name == "axial":
            axis_dim = 0  # Split along Z
        elif axis_name == "coronal":
            axis_dim = 1  # Split along Y
        else:
            raise ValueError("axis_name must be 'axial' or 'coronal'")

        num_slices = volume.shape[axis_dim]

        # Handle cases with very few slices by duplicating
        if num_slices < 3:
            mip = np.max(volume, axis=axis_dim)
            # Resize to target
            img = cv2.resize(mip, (self.IMG_SIZE, self.IMG_SIZE))
            # Normalize
            img = (img - (self.WL - self.WW / 2)) / self.WW
            img = np.clip(img, 0, 1)
            img_uint8 = (img * 255).astype(np.uint8)
            return np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

        # Define 3 overlapping slabs
        # 0-33%, 33-66%, 66-100% with 15% overlap
        slab_len = num_slices / 3
        overlap = int(slab_len * 0.15)

        p1 = int(num_slices / 3)
        p2 = int(2 * num_slices / 3)

        ranges = [
            (0, p1 + overlap),
            (max(0, p1 - overlap), p2 + overlap),
            (max(0, p2 - overlap), num_slices),
        ]

        channels = []
        for start, end in ranges:
            # Slice volume
            if axis_dim == 0:
                slab = volume[start:end, :, :]
            else:
                slab = volume[:, start:end, :]

            # Compute MIP
            if slab.shape[axis_dim] > 0:
                mip = np.max(slab, axis=axis_dim)
            else:
                mip = np.zeros_like(volume[0])  # Should not happen given logic

            # Resize to fixed resolution
            # Note: cv2.resize takes (width, height) -> (X, Y)
            # Input MIP might be (Y, X) or (Z, X).
            # We force everything to (224, 224)
            mip_resized = cv2.resize(mip, (self.IMG_SIZE, self.IMG_SIZE))

            # Apply Lung Windowing
            lower = self.WL - self.WW / 2
            mip_norm = (mip_resized - lower) / self.WW
            mip_norm = np.clip(mip_norm, 0, 1)

            channels.append((mip_norm * 255).astype(np.uint8))

        return np.stack(channels, axis=-1)

    def process_images(self, patient_id, dicom_dir, load_cached_data=True):
        """
        Main image processing pipeline for a patient.
        Returns (axial_image, coronal_image).
        """
        # Define cache paths
        cache_ax = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        cache_cor = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try Load
        if load_cached_data and os.path.exists(cache_ax) and os.path.exists(cache_cor):
            try:
                return np.load(cache_ax), np.load(cache_cor)
            except Exception:
                pass  # If load fails, recompute

        # 2. Compute
        # Construct full path to dicom directory
        # dicom_dir is relative, e.g., "train/ID..."
        full_path = os.path.join("./input", dicom_dir)

        volume = self.load_dicom_stack(full_path)

        img_ax = self.generate_tri_slab(volume, "axial")
        img_cor = self.generate_tri_slab(volume, "coronal")

        # 3. Save
        np.save(cache_ax, img_ax)
        np.save(cache_cor, img_cor)

        return img_ax, img_cor

    def prepare_tabular_features(self, train_df, val_df, test_df):
        """
        Processes tabular data.
        - Extracts baseline features for Train/Val to match Test format.
        - Computes time delta (Weeks_From_Baseline).
        - Normalizes numerical features and encodes categoricals.
        """

        def get_baseline_info(df):
            # If dataset already has baseline info (Test set), ensure columns exist and return
            if "Baseline_FVC" in df.columns:
                # Ensure Weeks_From_Baseline exists
                if "Weeks_From_Baseline" not in df.columns:
                    # For test, Weeks is Predict_Week, Baseline_Week is provided
                    df["Weeks_From_Baseline"] = df["Weeks"] - df["Baseline_Week"]
                return df.copy()

            # For Train/Val, we need to derive baseline from the history
            # Sort by Weeks to find the first visit
            df_sorted = df.sort_values(["Patient", "Weeks"])

            # Group by Patient and take the first record as baseline
            baseline = df_sorted.groupby("Patient").first().reset_index()

            # Rename columns to Baseline_...
            cols_map = {
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
                "Weeks": "Baseline_Week",
            }
            baseline = baseline[["Patient"] + list(cols_map.keys())].rename(
                columns=cols_map
            )

            # Merge baseline info back to the original dataframe
            df_merged = pd.merge(df, baseline, on="Patient", how="left")

            # Calculate time delta
            df_merged["Weeks_From_Baseline"] = (
                df_merged["Weeks"] - df_merged["Baseline_Week"]
            )

            return df_merged

        # 1. Align Dataframes
        # Ensure Test has 'Weeks' column for consistency
        if "Predict_Week" in test_df.columns:
            test_df = test_df.copy()
            test_df["Weeks"] = test_df["Predict_Week"]

        train_proc = get_baseline_info(train_df)
        val_proc = get_baseline_info(val_df)
        test_proc = get_baseline_info(test_df)

        # 2. Define Features
        num_cols = ["Baseline_Age", "Baseline_Percent"]
        cat_cols = ["Baseline_Sex", "Baseline_SmokingStatus"]

        # 3. Fit Scalers (on Train only)
        self.num_scaler.fit(train_proc[num_cols])
        self.cat_encoder.fit(train_proc[cat_cols])
        self.is_fitted = True

        # 4. Transform
        def transform(df):
            # Numerical
            nums = self.num_scaler.transform(df[num_cols])
            # Categorical
            cats = self.cat_encoder.transform(df[cat_cols])
            # Concatenate
            return np.hstack([nums, cats]).astype(np.float32)

        train_feats = transform(train_proc)
        val_feats = transform(val_proc)
        test_feats = transform(test_proc)

        return train_proc, train_feats, val_proc, val_feats, test_proc, test_feats
