import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from library.config import Config

# Attempt to import pydicom, define fallback if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class CTScanProcessor:
    """
    Handles loading, processing, and caching of CT scans.
    Implements the 'Lung-Extent Heuristic' to select 3 informative slices.
    """

    def __init__(self, img_size=Config.IMG_SIZE, num_slices=Config.NUM_SLICES):
        self.img_size = img_size
        self.num_slices = num_slices

    def _read_dicom_raw(self, filepath):
        """
        Fallback reader for DICOM files when pydicom is missing.
        Assumes 512x512 int16 uncompressed data at the end of the file.
        """
        try:
            file_size = os.path.getsize(filepath)
            # Standard DICOM image size: 512x512 * 2 bytes (int16) = 524288 bytes
            expected_data_size = 512 * 512 * 2

            if file_size < expected_data_size:
                return np.zeros((512, 512), dtype=np.int16)

            with open(filepath, "rb") as f:
                f.seek(-expected_data_size, 2)  # Seek to end minus data size
                buffer = f.read(expected_data_size)
                img = np.frombuffer(buffer, dtype=np.int16)
                img = img.reshape((512, 512))
                return img
        except Exception:
            return np.zeros((512, 512), dtype=np.int16)

    def read_slice(self, filepath):
        """Reads a single DICOM slice, converting to float32."""
        if HAS_PYDICOM:
            try:
                dcm = pydicom.dcmread(filepath)
                img = dcm.pixel_array.astype(np.float32)

                # Apply rescale slope/intercept to get Hounsfield Units (HU)
                slope = getattr(dcm, "RescaleSlope", 1)
                intercept = getattr(dcm, "RescaleIntercept", 0)
                img = img * slope + intercept
                return img
            except Exception:
                return self._read_dicom_raw(filepath).astype(np.float32)
        else:
            return self._read_dicom_raw(filepath).astype(np.float32)

    def process_patient(self, patient_id, image_dir_rel, load_cached=True):
        """
        Loads all slices for a patient, selects 3 based on lung area,
        resizes, and normalizes.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        if load_cached and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Recompute if load fails

        # Construct full path
        full_dir_path = os.path.join(Config.INPUT_DIR, image_dir_rel)

        if not os.path.exists(full_dir_path):
            return np.zeros(
                (self.num_slices, self.img_size, self.img_size), dtype=np.float32
            )

        files = [f for f in os.listdir(full_dir_path) if f.lower().endswith(".dcm")]

        # Sort by instance number (derived from filename, e.g., '10.dcm')
        try:
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            files.sort()

        if not files:
            return np.zeros(
                (self.num_slices, self.img_size, self.img_size), dtype=np.float32
            )

        # Read all slices to form a volume
        slices = []
        for f in files:
            img = self.read_slice(os.path.join(full_dir_path, f))
            slices.append(img)

        volume = np.stack(slices)  # (D, H, W)

        # --- Lung-Extent Heuristic ---
        # Calculate a proxy for "Lung Area" per slice.
        # Lung/Air is low density. We use a dynamic threshold based on volume mean.
        threshold = np.mean(volume)
        lung_areas = np.sum(volume < threshold, axis=(1, 2))

        # 1. Find Max Lung Area Slice (Anchor)
        idx_max = np.argmax(lung_areas)
        max_area = lung_areas[idx_max]

        # 2. Find Apical and Basal slices (~50% of max area)
        target_area = 0.5 * max_area

        # Apical: search indices < idx_max
        idx_apical = 0
        if idx_max > 0:
            candidates = lung_areas[:idx_max]
            idx_apical = np.argmin(np.abs(candidates - target_area))

        # Basal: search indices > idx_max
        idx_basal = len(files) - 1
        if idx_max < len(files) - 1:
            candidates = lung_areas[idx_max + 1 :]
            offset = np.argmin(np.abs(candidates - target_area))
            idx_basal = idx_max + 1 + offset

        selected_indices = [idx_apical, idx_max, idx_basal]

        # --- Resize and Normalize ---
        final_slices = []
        for idx in selected_indices:
            img = volume[idx]

            # Robust Min-Max Normalization to [0, 1]
            p1 = np.percentile(img, 1)
            p99 = np.percentile(img, 99)
            if p99 > p1:
                img = (img - p1) / (p99 - p1)
            else:
                img = img * 0
            img = np.clip(img, 0, 1)

            # Resize
            img_resized = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            final_slices.append(img_resized)

        output_volume = np.stack(final_slices).astype(np.float32)  # (3, 224, 224)

        # Save to cache
        try:
            np.save(cache_path, output_volume)
        except Exception:
            pass

        return output_volume


class TabularScaler:
    """
    Handles Z-score standardization for FVC, Weeks, Age, and Percent.
    Fits only on training data.
    """

    def __init__(self):
        self.stats = {}

    def fit(self, df):
        """Compute mean and std for numerical columns and FVC."""
        # Numerical columns: Weeks, Percent, Age
        for col in Config.NUMERICAL_COLS:
            self.stats[col] = {"mean": df[col].mean(), "std": df[col].std() + 1e-6}

        # Target FVC
        self.stats["FVC"] = {"mean": df["FVC"].mean(), "std": df["FVC"].std() + 1e-6}

    def transform(self, df):
        """Returns a dictionary of scaled arrays."""
        out = {}
        for col in Config.NUMERICAL_COLS:
            if col in df.columns:
                mean = self.stats[col]["mean"]
                std = self.stats[col]["std"]
                out[col] = (df[col].values - mean) / std

        if "FVC" in df.columns:
            mean = self.stats["FVC"]["mean"]
            std = self.stats["FVC"]["std"]
            out["FVC"] = (df["FVC"].values - mean) / std

        return out

    def scale_fvc(self, fvc_values):
        mean = self.stats["FVC"]["mean"]
        std = self.stats["FVC"]["std"]
        return (fvc_values - mean) / std

    def unscale_fvc(self, fvc_scaled):
        mean = self.stats["FVC"]["mean"]
        std = self.stats["FVC"]["std"]
        return fvc_scaled * std + mean

    def unscale_sigma(self, sigma_scaled):
        std = self.stats["FVC"]["std"]
        return sigma_scaled * std


class PulmonaryDataset(Dataset):
    """
    PyTorch Dataset for Lung FVC Prediction.
    Returns:
        - image: (3, H, W)
        - meta_cat: [Sex, SmokingStatus]
        - meta_num: [Age_scaled, Percent_scaled]
        - baseline_fvc_scaled
        - weeks_scaled
        - target_fvc_scaled
        - raw_fvc
    """

    def __init__(self, df, processor, scaler, baseline_lookup, mode="train"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.scaler = scaler
        self.baseline_lookup = baseline_lookup
        self.mode = mode

        # Pre-compute scaled tabular data
        self.scaled_data = self.scaler.transform(self.df)

        # Hardcoded encoders for consistency
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row[Config.ID_COL]

        # 1. Image
        image_path = row["image_path"]
        image = self.processor.process_patient(patient_id, image_path, load_cached=True)
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # 2. Metadata (Categorical)
        sex_idx = self.sex_map.get(row["Sex"], 0)
        smoke_idx = self.smoke_map.get(row["SmokingStatus"], 0)
        meta_cat = torch.tensor([sex_idx, smoke_idx], dtype=torch.long)

        # 3. Metadata (Numerical) - Age, Percent
        age_scaled = self.scaled_data["Age"][idx]
        percent_scaled = self.scaled_data["Percent"][idx]
        meta_num = torch.tensor([age_scaled, percent_scaled], dtype=torch.float32)

        # 4. Baseline FVC (Scaled)
        # Fallback to current FVC if baseline missing (should not happen with correct lookup)
        base_fvc = self.baseline_lookup.get(patient_id, row["FVC"])
        base_fvc_scaled = self.scaler.scale_fvc(base_fvc)

        # 5. Weeks (Scaled)
        weeks_scaled = self.scaled_data["Weeks"][idx]

        # 6. Target
        # In inference mode, FVC might be dummy, but we handle it safely
        raw_fvc = row["FVC"] if "FVC" in row else 0.0
        target_fvc_scaled = (
            self.scaled_data["FVC"][idx] if "FVC" in self.scaled_data else 0.0
        )

        return {
            "image": image_tensor,
            "meta_cat": meta_cat,
            "meta_num": meta_num,
            "baseline_fvc_scaled": torch.tensor(base_fvc_scaled, dtype=torch.float32),
            "weeks_scaled": torch.tensor(weeks_scaled, dtype=torch.float32),
            "target_fvc_scaled": torch.tensor(target_fvc_scaled, dtype=torch.float32),
            "raw_fvc": torch.tensor(raw_fvc, dtype=torch.float32),
            "patient_week": f"{patient_id}_{row['Weeks']}",
        }


def get_baseline_lookup(df):
    """
    Creates a dictionary {PatientID: BaselineFVC}.
    Baseline is defined as the FVC at the minimum week (earliest measurement).
    """
    lookup = {}
    for pid, group in df.groupby("Patient"):
        min_week_idx = group["Weeks"].idxmin()
        baseline_fvc = group.loc[min_week_idx, "FVC"]
        lookup[pid] = baseline_fvc
    return lookup


def prepare_inference_dataframe(sample_sub_path, test_meta_path):
    """
    Prepares the dataframe for submission by merging sample_submission
    with test metadata.
    """
    sub_df = pd.read_csv(sample_sub_path)
    test_meta = pd.read_csv(test_meta_path)

    # Split Patient_Week to get Patient and Weeks
    # Format: ID..._123
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge with metadata
    # test_meta contains 'Weeks' (baseline week) and 'FVC' (baseline FVC).
    # We rename them to avoid collision.
    test_meta = test_meta.rename(columns={"Weeks": "Base_Week", "FVC": "Base_FVC"})

    # Merge
    merged_df = sub_df.merge(test_meta, on="Patient", how="left")

    # For the Dataset class, we need 'FVC' column (target), even if dummy.
    # We also need 'Base_FVC' to be available for the lookup.
    # We'll use 'Base_FVC' as the 'FVC' column for the dataset to run without error,
    # but the actual baseline lookup will handle the logic.
    merged_df["FVC"] = merged_df["Base_FVC"]

    return merged_df
