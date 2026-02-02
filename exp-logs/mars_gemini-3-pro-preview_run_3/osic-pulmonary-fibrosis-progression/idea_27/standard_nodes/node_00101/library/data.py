import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from library.config import Config
from library.utils import TargetScaler


class CTPreprocessor:
    """
    Handles loading, windowing, and slice selection for CT scans.
    Implements caching to .npy files to improve training speed.
    """

    def __init__(self, cache_dir=Config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.target_size = (Config.IMG_SIZE, Config.IMG_SIZE)
        self.hu_level = Config.HU_LEVEL
        self.hu_width = Config.HU_WIDTH
        self.num_slices = Config.NUM_SLICES

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_hu_values(self, dcm_path):
        """Reads DICOM and converts to Hounsfield Units."""
        try:
            ds = pydicom.dcmread(dcm_path)
            pixel_array = ds.pixel_array.astype(np.float32)

            # Rescale to HU
            intercept = getattr(ds, "RescaleIntercept", -1024)
            slope = getattr(ds, "RescaleSlope", 1.0)
            img = pixel_array * slope + intercept

            # Get InstanceNumber for sorting
            inst_num = getattr(ds, "InstanceNumber", 0)
            return img, int(inst_num)
        except Exception:
            return None, None

    def window_image(self, img):
        """Applies radiological windowing (Lung Window)."""
        lower = self.hu_level - self.hu_width / 2
        upper = self.hu_level + self.hu_width / 2
        img = np.clip(img, lower, upper)
        # Normalize to [0, 1]
        img = (img - lower) / (upper - lower)
        return img

    def resize_image(self, img):
        """Resizes image to target dimensions."""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

    def select_slices(self, slices_data):
        """
        Selects Anchor (max lung area) + 2 boundary slices.
        slices_data: list of (img_hu, instance_number)
        """
        if not slices_data:
            return np.zeros((self.num_slices, *self.target_size), dtype=np.float32)

        # 1. Calculate approximate lung area for each slice
        # Lung tissue is typically between -1000 and -500 HU.
        # We use a threshold < -320 to identify air/lung tissue vs body.
        meta = []
        for img, num in slices_data:
            mask = (img > -1000) & (img < -320)
            area = np.sum(mask)
            meta.append({"area": area, "num": num, "img": img})

        # Sort by Instance Number to maintain spatial order
        meta.sort(key=lambda x: x["num"])

        areas = np.array([m["area"] for m in meta])
        max_area = np.max(areas)

        if max_area == 0:
            # Fallback: middle slices if no lung detected
            mid = len(meta) // 2
            selected_indices = [mid] * self.num_slices
        else:
            # Find Anchor (Max Area)
            anchor_idx = np.argmax(areas)

            # Find Candidates (> 50% max area)
            candidates = np.where(areas > 0.5 * max_area)[0]

            if len(candidates) < 3:
                # Not enough candidates, pad with anchor
                selected_indices = sorted(
                    list(set([candidates[0], anchor_idx, candidates[-1]]))
                )
                while len(selected_indices) < 3:
                    selected_indices.append(anchor_idx)
                selected_indices = sorted(selected_indices[:3])
            else:
                # Pick First Candidate (Top), Anchor, Last Candidate (Bottom)
                first = candidates[0]
                last = candidates[-1]

                # Use set to avoid duplicates if anchor is at boundary
                indices = sorted(list(set([first, anchor_idx, last])))

                if len(indices) == 3:
                    selected_indices = indices
                elif len(indices) == 2:
                    # Duplicate one. e.g. [A, B] -> [A, A, B]
                    selected_indices = [indices[0], indices[0], indices[1]]
                else:
                    selected_indices = [indices[0]] * 3

        # Extract, Window, Resize
        final_images = []
        for idx in selected_indices:
            img = meta[idx]["img"]
            img = self.window_image(img)
            img = self.resize_image(img)
            final_images.append(img)

        return np.array(final_images, dtype=np.float32)

    def process_patient(self, patient_id, image_dir_rel, load_cached_data=True):
        """
        Main processing function with caching.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Corrupt file, reprocess

        full_path = os.path.join(Config.INPUT_DIR, image_dir_rel)
        if not os.path.exists(full_path):
            return np.zeros((self.num_slices, *self.target_size), dtype=np.float32)

        files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
        slices_data = []
        for f in files:
            img, num = self.get_hu_values(os.path.join(full_path, f))
            if img is not None:
                slices_data.append((img, num))

        processed_vol = self.select_slices(slices_data)

        # Save to cache
        np.save(cache_path, processed_vol)

        return processed_vol


class LungDataset(Dataset):
    def __init__(
        self, df, ct_preprocessor, tabular_scalers=None, mode="train", baseline_df=None
    ):
        self.df = df.reset_index(drop=True)
        self.ct_preprocessor = ct_preprocessor
        self.tabular_scalers = tabular_scalers
        self.mode = mode

        # Build Baseline Lookup
        self.patient_baselines = {}

        # If explicit baseline_df is provided (e.g. for test set), use it.
        # Otherwise derive from self.df (training set).
        source_df = baseline_df if baseline_df is not None else self.df

        # Group by Patient to extract baseline stats (min Weeks)
        unique_pats = source_df["Patient"].unique()
        for pid in unique_pats:
            p_rows = source_df[source_df["Patient"] == pid]
            idx_min = p_rows["Weeks"].idxmin()
            base_row = p_rows.loc[idx_min]

            self.patient_baselines[pid] = {
                "FVC": base_row["FVC"],
                "Age": base_row["Age"],
                "Sex": base_row["Sex"],
                "SmokingStatus": base_row["SmokingStatus"],
                "Weeks": base_row["Weeks"],
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # Retrieve baseline info
        if pid not in self.patient_baselines:
            # Fallback (should not happen in valid flow)
            baseline = {
                "FVC": 2000,
                "Age": 65,
                "Sex": "Male",
                "SmokingStatus": "Ex-smoker",
                "Weeks": 0,
            }
        else:
            baseline = self.patient_baselines[pid]

        # 1. Image Processing
        img_path = row.get("image_path", f"test/{pid}")
        imgs = self.ct_preprocessor.process_patient(
            pid, img_path, load_cached_data=True
        )

        # 2. Clinical Features
        # Relative Time: Scale by 0.01
        t_rel = (row["Weeks"] - baseline["Weeks"]) * 0.01

        # Baseline FVC (Standardized)
        b_fvc = baseline["FVC"]
        if self.tabular_scalers and "fvc_scaler" in self.tabular_scalers:
            b_fvc = self.tabular_scalers["fvc_scaler"].transform(b_fvc)

        # Age (Standardized)
        age = baseline["Age"]
        if self.tabular_scalers and "age_scaler" in self.tabular_scalers:
            age = self.tabular_scalers["age_scaler"].transform(age)

        # Sex (Male=0, Female=1)
        sex = 0.0 if baseline["Sex"] == "Male" else 1.0

        # Smoking (Ex=0, Never=1, Current=2)
        s_str = baseline["SmokingStatus"]
        if s_str == "Ex-smoker":
            smoking = 0.0
        elif s_str == "Never smoked":
            smoking = 1.0
        else:
            smoking = 2.0

        # Feature Vector: [Baseline_FVC, Time, Age, Sex, Smoking]
        clin_vec = np.array([b_fvc, t_rel, age, sex, smoking], dtype=np.float32)

        # 3. Output Assembly
        res = {
            "image": torch.tensor(imgs, dtype=torch.float32),
            "clinical": torch.tensor(clin_vec, dtype=torch.float32),
        }

        if self.mode in ["train", "val"]:
            target_fvc = row["FVC"]
            # Scale target
            if self.tabular_scalers and "fvc_scaler" in self.tabular_scalers:
                target_fvc = self.tabular_scalers["fvc_scaler"].transform(target_fvc)
            res["target"] = torch.tensor(target_fvc, dtype=torch.float32)
        else:
            # For submission, return ID
            if "Patient_Week" in row:
                res["patient_week"] = row["Patient_Week"]
            else:
                res["patient_week"] = f"{pid}_{row['Weeks']}"

        return res


def get_dataloaders(debug=False):
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Initialize and Fit Scalers
    fvc_scaler = TargetScaler()
    fvc_scaler.fit(train_df["FVC"].values)

    age_scaler = TargetScaler()
    age_scaler.fit(train_df["Age"].values)

    scalers = {"fvc_scaler": fvc_scaler, "age_scaler": age_scaler}

    # Preprocessor
    ct_prep = CTPreprocessor()

    # Datasets
    train_ds = LungDataset(train_df, ct_prep, scalers, mode="train")
    val_ds = LungDataset(val_df, ct_prep, scalers, mode="val")

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

    return train_loader, val_loader, scalers


def get_test_dataloader(scalers):
    # Load data
    test_meta = pd.read_csv(Config.TEST_CSV)
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Prepare submission dataframe
    # Extract Patient and Weeks from Patient_Week string
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge with test_meta to get image_path for each patient
    merged_df = sub_df.merge(
        test_meta[["Patient", "image_path"]], on="Patient", how="left"
    )

    ct_prep = CTPreprocessor()

    # Pass test_meta as baseline_df so LungDataset can look up baseline stats correctly
    test_ds = LungDataset(
        merged_df, ct_prep, scalers, mode="test", baseline_df=test_meta
    )

    loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
