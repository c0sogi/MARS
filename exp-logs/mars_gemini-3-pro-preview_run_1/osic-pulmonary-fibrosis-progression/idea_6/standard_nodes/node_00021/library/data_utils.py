import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config

# Attempt to import pydicom (Standard for DICOM I/O)
try:
    import pydicom
except ImportError:
    # Fallback or error handling if pydicom is strictly unavailable
    # In this environment, we assume it is available as it is essential for the task.
    pydicom = None


class DataUtils:
    """
    Utility class for processing DICOM images and tabular clinical data.
    Implements caching to optimize runtime.
    """

    @staticmethod
    def load_scan(path):
        """
        Loads all DICOM files from a directory and sorts them by InstanceNumber.
        """
        if not os.path.exists(path):
            return []

        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            return []

        scans = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                scans.append(ds)
            except Exception:
                continue

        # Sort by InstanceNumber to ensure correct Z-ordering
        scans.sort(key=lambda x: int(x.InstanceNumber))
        return scans

    @staticmethod
    def get_pixels_hu(scans):
        """
        Converts raw DICOM pixel_array to Hounsfield Units (HU).
        """
        if not scans:
            return np.zeros((1, 512, 512), dtype=np.int16)

        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0 (Air)
        # The intercept is usually -1024, so air is approximately -1000
        # Padding is often -2000
        image[image == -2000] = 0

        # Convert to Hounsfield Units (HU)
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    @staticmethod
    def generate_tri_slab(volume, view="axial", img_size=224):
        """
        Generates a 3-channel image using overlapping Tri-Slab Maximum Intensity Projections (MIPs).

        Args:
            volume (np.array): 3D array (Z, Y, X)
            view (str): 'axial' or 'coronal'
            img_size (int): Output spatial resolution

        Returns:
            np.array: (img_size, img_size, 3) normalized to [0, 1]
        """
        # Orient volume based on view
        # Axial: (Z, Y, X) -> Slice along Z (dim 0)
        # Coronal: (Y, Z, X) -> Slice along Y (dim 1)
        if view == "coronal":
            # Transpose to make Y the slicing dimension (0)
            # Original: Z(0), Y(1), X(2) -> Y(1), Z(0), X(2)
            vol = volume.transpose(1, 0, 2)
        else:
            vol = volume

        depth = vol.shape[0]

        # Handle cases with very few slices
        if depth < 3:
            mip = np.max(vol, axis=0)
            mips = [mip, mip, mip]
        else:
            # Define 3 overlapping slabs covering the volume
            # Overlap factor: 10% of total depth
            overlap = int(depth * 0.10)
            chunk = depth // 3

            # Slab indices
            s1_start, s1_end = 0, chunk + overlap
            s2_start, s2_end = chunk - overlap, 2 * chunk + overlap
            s3_start, s3_end = 2 * chunk - overlap, depth

            # Clamp indices
            s1_end = min(s1_end, depth)
            s2_start = max(s2_start, 0)
            s2_end = min(s2_end, depth)
            s3_start = max(s3_start, 0)

            # Compute MIPs
            mip1 = np.max(vol[s1_start:s1_end], axis=0)
            mip2 = np.max(vol[s2_start:s2_end], axis=0)
            mip3 = np.max(vol[s3_start:s3_end], axis=0)

            mips = [mip1, mip2, mip3]

        # Process each channel (Resize, Window, Normalize)
        channels = []
        for mip in mips:
            # Resize
            if mip.shape[0] != img_size or mip.shape[1] != img_size:
                mip = cv2.resize(
                    mip, (img_size, img_size), interpolation=cv2.INTER_AREA
                )

            # Windowing (Lung Window: -1000 to 400 HU)
            mip = np.clip(mip, -1000, 400)

            # Normalize to 0-1
            mip = (mip - (-1000)) / (400 - (-1000))

            channels.append(mip)

        # Stack to (H, W, 3)
        img = np.stack(channels, axis=-1)
        return img.astype(np.float32)

    @staticmethod
    def preprocess_and_cache_images(df, cache_dir, load_cached_data=True):
        """
        Iterates over unique patients in the dataframe, processes their CT scans,
        and caches the result as .npy files.

        Returns:
            dict: Mapping from PatientID to cached file path.
        """
        os.makedirs(cache_dir, exist_ok=True)

        patient_ids = df["Patient"].unique()
        patient_path_map = {}

        # print(f"Processing images for {len(patient_ids)} patients...")

        for pid in patient_ids:
            save_path = os.path.join(cache_dir, f"{pid}.npy")
            patient_path_map[pid] = save_path

            if load_cached_data and os.path.exists(save_path):
                continue

            # Process from scratch
            try:
                # Find directory
                rel_dir = df[df["Patient"] == pid]["dicom_dir"].iloc[0]
                full_dir = os.path.join(Config.input_root, rel_dir)

                scans = DataUtils.load_scan(full_dir)

                if not scans:
                    # Create zero arrays if no scans found
                    img_data = {
                        "axial": np.zeros(
                            (Config.img_size, Config.img_size, 3), dtype=np.float32
                        ),
                        "coronal": np.zeros(
                            (Config.img_size, Config.img_size, 3), dtype=np.float32
                        ),
                    }
                else:
                    vol = DataUtils.get_pixels_hu(scans)

                    # Generate views
                    axial = DataUtils.generate_tri_slab(vol, "axial", Config.img_size)
                    coronal = DataUtils.generate_tri_slab(
                        vol, "coronal", Config.img_size
                    )

                    img_data = {"axial": axial, "coronal": coronal}
            except Exception as e:
                # Fallback for corrupt data
                img_data = {
                    "axial": np.zeros(
                        (Config.img_size, Config.img_size, 3), dtype=np.float32
                    ),
                    "coronal": np.zeros(
                        (Config.img_size, Config.img_size, 3), dtype=np.float32
                    ),
                }

            # Save to cache
            np.save(save_path, img_data)

        return patient_path_map

    @staticmethod
    def process_tabular_features(df, cache_dir, mode="train"):
        """
        Encodes and normalizes tabular features: Age, Sex, SmokingStatus, Percent.
        Handles state saving/loading for consistent scaling between train and test.
        """
        data = df.copy()

        # Normalize column names for Test set (which has Baseline_ prefix)
        if "Baseline_Age" in data.columns:
            data["Age"] = data["Baseline_Age"]
            data["Sex"] = data["Baseline_Sex"]
            data["SmokingStatus"] = data["Baseline_SmokingStatus"]
            data["Percent"] = data["Baseline_Percent"]

        # Paths for saved encoders
        sex_enc_path = os.path.join(cache_dir, "sex_enc.npy")
        smoke_enc_path = os.path.join(cache_dir, "smoke_enc.npy")
        scaler_path = os.path.join(cache_dir, "scaler_params.npy")  # [mean, scale]

        sex_enc = LabelEncoder()
        smoke_enc = LabelEncoder()
        scaler = StandardScaler()

        if mode == "train":
            # Fit and Transform
            data["Sex_Enc"] = sex_enc.fit_transform(data["Sex"])
            data["Smoke_Enc"] = smoke_enc.fit_transform(data["SmokingStatus"])

            num_features = data[["Age", "Percent"]].values
            scaled_features = scaler.fit_transform(num_features)

            # Save state
            np.save(sex_enc_path, sex_enc.classes_)
            np.save(smoke_enc_path, smoke_enc.classes_)
            np.save(scaler_path, np.array([scaler.mean_, scaler.scale_]))

        else:
            # Load state and Transform
            try:
                sex_classes = np.load(sex_enc_path, allow_pickle=True)
                smoke_classes = np.load(smoke_enc_path, allow_pickle=True)
                scaler_params = np.load(scaler_path, allow_pickle=True)

                sex_enc.classes_ = sex_classes
                smoke_enc.classes_ = smoke_classes
                scaler.mean_ = scaler_params[0]
                scaler.scale_ = scaler_params[1]

                # Transform
                # Map unknown labels to a default if necessary (simple approach here)
                data["Sex_Enc"] = sex_enc.transform(data["Sex"])
                data["Smoke_Enc"] = smoke_enc.transform(data["SmokingStatus"])

                num_features = data[["Age", "Percent"]].values
                scaled_features = scaler.transform(num_features)

            except FileNotFoundError:
                # Fallback (should not happen if train runs first)
                data["Sex_Enc"] = 0
                data["Smoke_Enc"] = 0
                scaled_features = data[["Age", "Percent"]].values

        # Construct final feature matrix
        # Columns: [Age (scaled), Sex (enc), Smoking (enc), Percent (scaled)]
        processed = np.column_stack(
            [
                scaled_features[:, 0],  # Age
                data["Sex_Enc"].values,
                data["Smoke_Enc"].values,
                scaled_features[:, 1],  # Percent
            ]
        ).astype(np.float32)

        return processed

    @staticmethod
    def prepare_dataset(df, cache_dir, mode="train", load_cached_data=True):
        """
        Orchestrates the data preparation pipeline.

        Args:
            df (pd.DataFrame): Input dataframe.
            cache_dir (str): Directory for caching.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use existing cache.

        Returns:
            dict: Contains tensors for 'meta', 'targets', 'time_delta', 'base_fvc'
                  and a list of 'img_paths'.
        """
        # 1. Process Images (Unique Patients)
        patient_map = DataUtils.preprocess_and_cache_images(
            df, cache_dir, load_cached_data
        )

        # 2. Process Tabular Data (All Rows)
        tab_features = DataUtils.process_tabular_features(df, cache_dir, mode)
        meta_tensor = torch.tensor(tab_features, dtype=torch.float32)

        # 3. Map Image Paths to Rows
        img_paths = [patient_map[pid] for pid in df["Patient"].values]

        # 4. Prepare Targets and Time Deltas
        if mode in ["train", "val"]:
            # We need to establish a Baseline FVC for the training set to simulate the task
            # (Predicting future FVC based on initial).
            # We group by patient and take the first chronological visit as baseline.
            if "Baseline_FVC" not in df.columns:
                df_sorted = df.sort_values(["Patient", "Weeks"])
                baseline_df = df_sorted.groupby("Patient").first().reset_index()

                base_fvc_map = dict(zip(baseline_df["Patient"], baseline_df["FVC"]))
                base_week_map = dict(zip(baseline_df["Patient"], baseline_df["Weeks"]))

                df["Baseline_FVC"] = df["Patient"].map(base_fvc_map)
                df["Baseline_Week"] = df["Patient"].map(base_week_map)

            targets = torch.tensor(df["FVC"].values, dtype=torch.float32).view(-1, 1)
            base_fvc = torch.tensor(
                df["Baseline_FVC"].values, dtype=torch.float32
            ).view(-1, 1)

            # Time Delta = Current Week - Baseline Week
            time_delta = torch.tensor(
                df["Weeks"].values - df["Baseline_Week"].values, dtype=torch.float32
            ).view(-1, 1)

        else:
            # Test Mode
            # df already has Baseline_FVC, Predict_Week, Baseline_Week
            targets = torch.zeros((len(df), 1))  # Dummy targets
            base_fvc = torch.tensor(
                df["Baseline_FVC"].values, dtype=torch.float32
            ).view(-1, 1)
            time_delta = torch.tensor(
                df["Predict_Week"].values - df["Baseline_Week"].values,
                dtype=torch.float32,
            ).view(-1, 1)

        return {
            "meta": meta_tensor,
            "img_paths": img_paths,
            "targets": targets,
            "time_delta": time_delta,
            "base_fvc": base_fvc,
            "raw_df": df,
        }
