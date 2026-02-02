import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config
from library.utils import seed_everything


def get_img_custom(path):
    """
    Loads DICOMs, applies lung windowing, and selects 3 specific slices
    (Anchor + 2 Boundaries) based on lung area.
    """
    if not os.path.exists(path):
        # Return a black volume if path doesn't exist (robustness)
        return np.zeros(
            (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )

    dcm_files = []
    for f in os.listdir(path):
        if f.lower().endswith(".dcm"):
            try:
                dcm_files.append(pydicom.dcmread(os.path.join(path, f)))
            except:
                continue

    # Sort by InstanceNumber (Z-position)
    dcm_files.sort(key=lambda x: int(x.InstanceNumber))

    if len(dcm_files) == 0:
        return np.zeros(
            (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )

    # Convert to HU and apply Windowing
    # Window: Level -600, Width 1500 => Range [-1350, 150]
    L = Config.WINDOW_LEVEL
    W = Config.WINDOW_WIDTH
    min_hu = L - W // 2
    max_hu = L + W // 2

    slices = []
    areas = []

    for dcm in dcm_files:
        # Extract pixel array and convert to HU
        try:
            img = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"Warning: Skipping corrupt DICOM slice in {path}: {e}")
            continue

        intercept = getattr(dcm, "RescaleIntercept", 0)
        slope = getattr(dcm, "RescaleSlope", 1)
        img = img * slope + intercept

        # Calculate Lung Area (heuristic: pixels < -400)
        # Cite solution_lesson_node_00090: Avoid excluding valid air voxels (<-1000) in lung mask.
        lung_pixels = (img < -400).sum()
        areas.append(lung_pixels)

        # Apply Windowing
        img = np.clip(img, min_hu, max_hu)
        # Normalize to [0, 1]
        img = (img - min_hu) / (max_hu - min_hu)

        # Resize
        if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

        slices.append(img)

    if len(slices) == 0:
        return np.zeros(
            (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )

    slices = np.array(slices)
    areas = np.array(areas)

    # Slice Selection Logic
    if len(slices) < Config.SLICES_PER_PATIENT:
        # Padding if not enough slices
        diff = Config.SLICES_PER_PATIENT - len(slices)
        padding = np.zeros((diff, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
        final_volume = np.concatenate([slices, padding], axis=0)
    else:
        # 1. Find Anchor (Max Area)
        max_area_idx = np.argmax(areas)
        max_area = areas[max_area_idx]

        # 2. Find Boundaries (Area > 50% of Max)
        threshold = 0.5 * max_area
        valid_indices = np.where(areas >= threshold)[0]

        if len(valid_indices) > 0:
            min_bound_idx = valid_indices[0]
            max_bound_idx = valid_indices[-1]
        else:
            min_bound_idx = max_area_idx
            max_bound_idx = max_area_idx

        # Select indices: [Top Boundary, Anchor, Bottom Boundary]
        # We sort them spatially to maintain anatomical consistency if desired,
        # or keep them as semantic channels.
        # The idea implies "Anchor ... and two boundary slices".
        # Let's stack them spatially: min_bound -> max_area -> max_bound
        # However, to ensure distinct channels for the CNN, we pick these 3 specific indices.

        selected_indices = sorted(
            list(set([min_bound_idx, max_area_idx, max_bound_idx]))
        )

        # If we have fewer than 3 unique indices (e.g. max is the only valid one), duplicate
        while len(selected_indices) < 3:
            selected_indices.append(selected_indices[-1])

        # If we have more than 3 (rare, but if logic above added too many), take boundaries and center
        if len(selected_indices) > 3:
            # This shouldn't happen with the set logic above unless we added logic I removed.
            # But just in case, take first, middle, last.
            selected_indices = [selected_indices[0], max_area_idx, selected_indices[-1]]

        final_volume = slices[selected_indices[:3]]

    return final_volume


def cache_data(patients, input_dir, cache_dir):
    """
    Processes and caches image data for a list of patients.
    """
    print(f"Checking cache for {len(patients)} patients...")

    processed_count = 0
    for patient in patients:
        save_path = os.path.join(cache_dir, f"{patient}.npy")

        if os.path.exists(save_path):
            continue

        # Determine source path (train or test)
        # We check both locations
        train_path = os.path.join(input_dir, "train", patient)
        test_path = os.path.join(input_dir, "test", patient)

        if os.path.exists(train_path):
            img_path = train_path
        elif os.path.exists(test_path):
            img_path = test_path
        else:
            # Should not happen based on metadata
            print(f"Warning: Image directory not found for {patient}")
            dummy = np.zeros(
                (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32,
            )
            np.save(save_path, dummy)
            continue

        # Process
        volume = get_img_custom(img_path)
        np.save(save_path, volume)
        processed_count += 1

    if processed_count > 0:
        print(f"Processed and cached {processed_count} new patients.")
    else:
        print("All patients already cached.")


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", scalers=None):
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.mode = mode
        self.scalers = scalers

        # Pre-compute features
        self._prepare_tabular_data()

    def _prepare_tabular_data(self):
        # 1. Baseline Extraction
        # We need to find the baseline FVC and Week for each patient.
        # In this dataset, 'Weeks' are relative to baseline CT.
        # However, we need the initial FVC measurement.
        # For train/val, we have history. For test, we only have the first row.

        # We group by patient to find the baseline entry (min weeks)
        # Note: In the provided metadata, train.csv has all visits.
        # We assume the dataframe passed here has the necessary columns.

        # Create a baseline lookup
        # For training data, the baseline is the entry with min(Weeks).
        # We create a temporary dataframe to merge baseline info back.
        baseline_df = (
            self.df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "FVC", "Weeks"]].rename(
            columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
        )

        # Merge baseline info
        self.df = self.df.merge(baseline_df, on="Patient", how="left")

        # 2. Feature Engineering
        # Relative Time
        self.df["Relative_Time"] = (
            self.df["Weeks"] - self.df["Baseline_Week"]
        ) * Config.TIME_SCALE

        # Ordinal Encoding for SmokingStatus
        # 0: Ex-smoker, 1: Never smoked, 2: Currently smokes (Example mapping, we use sklearn)
        if self.scalers and "smoking_encoder" in self.scalers:
            enc = self.scalers["smoking_encoder"]
            # Handle unknown categories if any (though dataset is clean)
            smoking_reshaped = self.df["SmokingStatus"].values.reshape(-1, 1)
            self.df["Smoking_Code"] = enc.transform(smoking_reshaped).flatten()
        else:
            # Fallback or manual map if no scaler provided (should not happen in strict pipeline)
            # Just manual map for safety if scaler is missing
            mapping = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
            self.df["Smoking_Code"] = self.df["SmokingStatus"].map(mapping).fillna(0)

        # Sex Encoding
        self.df["Sex_Code"] = self.df["Sex"].apply(lambda x: 0 if x == "Male" else 1)

        # 3. Standardization (Age, Baseline FVC)
        if self.scalers and "standard_scaler" in self.scalers:
            scaler = self.scalers["standard_scaler"]
            cols_to_scale = ["Age", "Baseline_FVC"]
            scaled_vals = scaler.transform(self.df[cols_to_scale])
            self.df["Age_Scaled"] = scaled_vals[:, 0]
            self.df["Baseline_FVC_Scaled"] = scaled_vals[:, 1]
        else:
            # Should not happen in training
            self.df["Age_Scaled"] = self.df["Age"]
            self.df["Baseline_FVC_Scaled"] = self.df["Baseline_FVC"]

        # 4. Target Standardization
        if self.mode != "test" and self.scalers and "target_scaler" in self.scalers:
            target_scaler = self.scalers["target_scaler"]
            self.df["FVC_Target_Scaled"] = target_scaler.transform(self.df[["FVC"]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Load Image from Cache
        img_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
        if os.path.exists(img_path):
            image = np.load(img_path)
        else:
            # Fallback if cache missing (should be handled by cache_data)
            image = np.zeros(
                (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32,
            )

        # Tabular Features
        # [Baseline FVC, Relative Time, Age, Sex, SmokingStatus]
        # Note: Baseline FVC and Age are scaled. Relative Time is scaled. Sex/Smoking are codes.
        tabular = np.array(
            [
                row["Baseline_FVC_Scaled"],
                row["Relative_Time"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoking_Code"],
            ],
            dtype=np.float32,
        )

        data = {
            "image": torch.tensor(image, dtype=torch.float32),
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "patient_week": f"{patient_id}_{row['Weeks']}",
        }

        if self.mode != "test":
            data["fvc_target"] = torch.tensor(
                row["FVC_Target_Scaled"], dtype=torch.float32
            )
            # We don't have ground truth sigma, model learns it.
            # But we pass raw FVC for metric calculation if needed (though metric usually calc outside)
            data["fvc_raw"] = torch.tensor(row["FVC"], dtype=torch.float32)

        return data


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles caching, scaler fitting, and dataset creation.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Cache Images
    # Collect all unique patients
    all_patients = pd.concat(
        [train_df["Patient"], val_df["Patient"], test_df["Patient"]]
    ).unique()

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if not load_cached_data:
        # Clear existing cache if requested (optional, but safer to just overwrite or skip)
        pass

    # Trigger caching
    cache_data(all_patients, Config.INPUT_DIR, Config.CACHE_DIR)

    # 3. Fit Scalers (on Training Data ONLY)
    scalers = {}

    # Ordinal Encoder for Smoking
    # We fit on all possible values to ensure coverage, or just train if we are strict.
    # Given the small categories, fitting on known categories is safe.
    smoking_encoder = OrdinalEncoder(
        categories=[["Ex-smoker", "Never smoked", "Currently smokes"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    # We need to fit it to get the internal state ready, even if categories are fixed
    smoking_encoder.fit(
        np.array([["Ex-smoker"], ["Never smoked"], ["Currently smokes"]])
    )
    scalers["smoking_encoder"] = smoking_encoder

    # Standard Scaler for Inputs (Age, Baseline FVC)
    # We need to extract the baseline FVC for training patients first
    train_baseline_df = (
        train_df.sort_values(["Patient", "Weeks"])
        .groupby("Patient")
        .first()
        .reset_index()
    )
    train_baseline_df = train_baseline_df.rename(columns={"FVC": "Baseline_FVC"})

    # We also need Age from the main df (Age is constant per patient usually, but let's take from baseline df)

    input_scaler = StandardScaler()
    input_features = train_baseline_df[["Age", "Baseline_FVC"]].values
    input_scaler.fit(input_features)
    scalers["standard_scaler"] = input_scaler

    # Standard Scaler for Target (FVC)
    # Fit on all training FVC measurements
    target_scaler = StandardScaler()
    target_scaler.fit(train_df[["FVC"]].values)
    scalers["target_scaler"] = target_scaler

    # 4. Create Datasets
    train_dataset = OSICDataset(
        train_df, Config.CACHE_DIR, mode="train", scalers=scalers
    )
    val_dataset = OSICDataset(val_df, Config.CACHE_DIR, mode="val", scalers=scalers)
    test_dataset = OSICDataset(test_df, Config.CACHE_DIR, mode="test", scalers=scalers)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Return scalers as well for inverse transform later
    return train_loader, val_loader, test_loader, scalers
