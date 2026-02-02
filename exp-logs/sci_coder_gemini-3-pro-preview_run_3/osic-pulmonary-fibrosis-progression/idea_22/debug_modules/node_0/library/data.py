import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config, seed_everything


class TabularPreprocessor:
    """
    Handles preprocessing of tabular data:
    1. Identifies baseline features (FVC, Percent) for each patient.
    2. Computes relative time.
    3. Encodes categorical features.
    4. Standardizes numerical features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False

        # Mappings
        self.sex_map = {"Male": 1, "Female": 0}
        self.smoke_map = {"Ex-smoker": 1, "Never smoked": 0, "Currently smokes": 2}

    def _get_baseline(self, df):
        """
        Extracts baseline FVC, Percent, and Week for each patient.
        Baseline is defined as the visit closest to Week 0.
        """
        # Create a copy to avoid modifying original
        temp_df = df.copy()
        temp_df["abs_weeks"] = temp_df["Weeks"].abs()

        # Sort by patient and distance to week 0
        temp_df = temp_df.sort_values(["Patient", "abs_weeks"])

        # Drop duplicates to keep the first (closest to 0) for each patient
        baseline_df = temp_df.drop_duplicates("Patient", keep="first")

        return baseline_df[["Patient", "FVC", "Percent", "Weeks"]].rename(
            columns={"FVC": "Base_FVC", "Percent": "Base_Percent", "Weeks": "Base_Week"}
        )

    def fit(self, train_df):
        """
        Fits the scalers on the training data.
        """
        # Prepare training features
        df = train_df.copy()
        baseline_df = self._get_baseline(df)
        df = df.merge(baseline_df, on="Patient", how="left")

        # Calculate Relative Time
        df["Rel_Week"] = (df["Weeks"] - df["Base_Week"]) * Config.TIME_SCALE

        # Encode Categoricals
        df["Sex_Code"] = df["Sex"].map(self.sex_map).fillna(0)
        df["Smoke_Code"] = df["SmokingStatus"].map(self.smoke_map).fillna(0)

        # Select features for scaling
        # Features: [Base_FVC, Base_Percent, Age]
        features_to_fit = df[["Base_FVC", "Base_Percent", "Age"]].values
        self.scaler.fit(features_to_fit)

        # Fit target scaler
        self.target_scaler.fit(df[["FVC"]].values)

        self.is_fitted = True

    def transform(self, df, mode="train"):
        """
        Transforms the dataframe into input tensors.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df = df.copy()

        # Merge baseline info
        if mode == "test":
            # In test set, the single row per patient is the baseline
            df["Base_FVC"] = df["FVC"]
            df["Base_Percent"] = df["Percent"]
            df["Base_Week"] = df["Weeks"]
        else:
            # For train/val, derive baseline from history
            baseline_df = self._get_baseline(df)
            df = df.merge(baseline_df, on="Patient", how="left")

        # Calculate Relative Time
        df["Rel_Week"] = (df["Weeks"] - df["Base_Week"]) * Config.TIME_SCALE

        # Encode
        df["Sex_Code"] = df["Sex"].map(self.sex_map).fillna(0)
        df["Smoke_Code"] = df["SmokingStatus"].map(self.smoke_map).fillna(0)

        # Standardize numericals
        num_feats = df[["Base_FVC", "Base_Percent", "Age"]].values
        num_feats_scaled = self.scaler.transform(num_feats)

        # Construct final feature matrix
        # Order: [Base_FVC, Base_Percent, Rel_Week, Age, Sex, Smoking]
        base_fvc = num_feats_scaled[:, 0]
        base_pct = num_feats_scaled[:, 1]
        age = num_feats_scaled[:, 2]
        rel_week = df["Rel_Week"].values
        sex = df["Sex_Code"].values
        smoke = df["Smoke_Code"].values

        X = np.stack([base_fvc, base_pct, rel_week, age, sex, smoke], axis=1)

        # Handle Target
        y = None
        if "FVC" in df.columns:
            y = df[["FVC"]].values

        return X.astype(np.float32), y, df["Patient"].values, df["Weeks"].values

    def transform_target(self, y):
        return self.target_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(y_scaled)


class LungDataset(Dataset):
    def __init__(self, df, preprocessor, mode="train", cache_dir=Config.CACHE_DIR):
        self.df = df
        self.preprocessor = preprocessor
        self.mode = mode
        self.cache_dir = cache_dir

        # Process tabular data
        self.X, self.y, self.patients, self.weeks = self.preprocessor.transform(
            df, mode
        )

        # If train/val, we pre-calculate scaled targets
        self.y_scaled = None
        if self.y is not None:
            self.y_scaled = self.preprocessor.transform_target(self.y)

    def __len__(self):
        return len(self.df)

    def _load_image(self, patient_id):
        """
        Loads image from cache or processes from DICOM.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing if load fails

        # Process DICOM
        img = self._process_dicom(patient_id)

        # Save to cache
        np.save(cache_path, img)
        return img

    def _process_dicom(self, patient_id):
        """
        Reads DICOMs, selects adaptive slices, resizes, and normalizes.
        """
        # Determine directory based on mode (train/val use 'train' dir, test uses 'test' dir)
        # Note: val dataset uses mode='train' logic for tabular, but images are in 'train' folder
        folder_name = "test" if self.mode == "test" else "train"
        patient_dir = os.path.join(Config.DICOM_ROOT, folder_name, patient_id)

        if not os.path.exists(patient_dir):
            # Return zero tensor if not found (should not happen)
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        files = [f for f in os.listdir(patient_dir) if f.endswith(".dcm")]
        if not files:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Sort files numerically
        files.sort(
            key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0
        )

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(os.path.join(patient_dir, f))
                # Convert to HU
                slope = float(dcm.RescaleSlope) if hasattr(dcm, "RescaleSlope") else 1.0
                intercept = (
                    float(dcm.RescaleIntercept)
                    if hasattr(dcm, "RescaleIntercept")
                    else 0.0
                )
                img = dcm.pixel_array.astype(np.float32) * slope + intercept
                slices.append(img)
            except:
                continue

        if not slices:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Stack to (N, H, W)
        vol = np.stack(slices)

        # Adaptive Slice Selection
        # Calculate lung area (approx) per slice using threshold < -400 HU
        areas = np.sum(vol < -400, axis=(1, 2))

        if len(areas) == 0:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Find Max Area Slice
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        # Find boundaries (50% of max area)
        threshold = max_area * Config.SLICE_THRESHOLD

        # Search below
        low_idx = max_idx
        for i in range(max_idx, -1, -1):
            if areas[i] < threshold:
                low_idx = i
                break

        # Search above
        high_idx = max_idx
        for i in range(max_idx, len(slices)):
            if areas[i] < threshold:
                high_idx = i
                break

        # Select indices: Low, Max, High
        selected_indices = [low_idx, max_idx, high_idx]
        selected_indices.sort()

        # Extract slices
        final_slices = []
        for idx in selected_indices:
            sl = vol[idx]

            # Windowing (Lung Window: W=1500, L=-600 -> Min=-1350, Max=150)
            sl = np.clip(sl, -1350, 150)

            # Normalize to 0-1
            sl = (sl - (-1350)) / (150 - (-1350))

            # Resize
            sl = cv2.resize(sl, (Config.IMG_SIZE, Config.IMG_SIZE))
            final_slices.append(sl)

        return np.stack(final_slices).astype(np.float32)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]

        # Load Image (C, H, W)
        img = self._load_image(patient_id)
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Load Tabular
        tab = self.X[idx]
        tab_tensor = torch.tensor(tab, dtype=torch.float32)

        # Load Target
        if self.mode != "test":
            target = self.y[idx]  # Raw FVC
            target_scaled = self.y_scaled[idx]  # Scaled FVC

            return {
                "image": img_tensor,
                "tabular": tab_tensor,
                "target": torch.tensor(target, dtype=torch.float32).squeeze(),
                "target_scaled": torch.tensor(
                    target_scaled, dtype=torch.float32
                ).squeeze(),
                "patient_week": f"{patient_id}_{self.weeks[idx]}",
            }
        else:
            return {
                "image": img_tensor,
                "tabular": tab_tensor,
                "patient_week": f"{patient_id}_{self.weeks[idx]}",
            }


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SIZE]
        # Keep test small if needed, but usually test is small anyway

    # Initialize Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    # Create Datasets
    # Note: Val dataset uses mode='train' because it shares the same structure (history, targets)
    train_ds = LungDataset(train_df, preprocessor, mode="train")
    val_ds = LungDataset(val_df, preprocessor, mode="train")
    test_ds = LungDataset(test_df, preprocessor, mode="test")

    # Create Loaders
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

    return train_loader, val_loader, test_loader, preprocessor
