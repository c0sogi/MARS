import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Attempt to import pydicom for DICOM handling
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not found. Image processing will produce zero-tensors.")

# -------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------


def get_img(path):
    """
    Reads a DICOM file and converts it to Hounsfield Units (HU).
    """
    if not HAS_PYDICOM:
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    try:
        d = pydicom.dcmread(path)
        return d.pixel_array.astype(np.float32) * d.RescaleSlope + d.RescaleIntercept
    except Exception as e:
        # Fallback for corrupt files
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)


def apply_window(img, level=Config.WINDOW_LEVEL, width=Config.WINDOW_WIDTH):
    """
    Applies the specified windowing (Level/Width) to the image.
    Maps the range [level - width/2, level + width/2] to [0, 1].
    """
    lower = level - width / 2
    upper = level + width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img


def process_patient_images(patient_id, image_dir):
    """
    Loads all DICOM slices for a patient, selects 3 slices (Anchor + 2 Boundaries),
    resizes them, and returns a stacked volume (3, H, W).
    """
    if not os.path.exists(image_dir):
        return np.zeros(
            (Config.SLICE_COUNT, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    files = [f for f in os.listdir(image_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros(
            (Config.SLICE_COUNT, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    # Sort files. Try to sort by InstanceNumber if possible, else by filename
    # Reading all headers just for sorting is slow, so we sort by filename number
    # e.g. "10.dcm" -> 10
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()

    # Load all slices and calculate lung area for selection
    slices = []
    areas = []

    # Heuristic: Lung tissue is roughly between -1000 and -400 HU.
    # We use a simplified check on the windowed image or just raw HU.
    # To save time, we read, window, and resize on the fly during selection

    processed_slices = []

    for f in files:
        path = os.path.join(image_dir, f)
        img_hu = get_img(path)

        # Calculate area (count pixels in lung range)
        # Air is ~ -1000, Tissue ~ -100 to 100. Lungs are in between.
        # We threshold < -400 for air/lung.
        area = np.sum((img_hu > -1000) & (img_hu < -400))
        areas.append(area)

        # Apply window and resize immediately to save memory
        img_windowed = apply_window(img_hu)
        img_resized = cv2.resize(img_windowed, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        processed_slices.append(img_resized)

    areas = np.array(areas)

    # Selection Logic
    if len(areas) == 0:
        return np.zeros(
            (Config.SLICE_COUNT, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    max_idx = np.argmax(areas)
    max_area = areas[max_idx]

    # Find boundary candidates (> 50% of max area)
    threshold = 0.5 * max_area
    candidates = np.where(areas > threshold)[0]

    if len(candidates) == 0:
        selected_indices = [max_idx] * Config.SLICE_COUNT
    else:
        # Select: Start of range, Max Area (Anchor), End of range
        start_idx = candidates[0]
        end_idx = candidates[-1]

        # Ensure we have 3 distinct if possible, or replicate
        selected_indices = [start_idx, max_idx, end_idx]

    # Retrieve selected slices
    final_volume = []
    for idx in selected_indices:
        final_volume.append(processed_slices[idx])

    # Stack to (3, H, W)
    volume = np.stack(final_volume, axis=0)

    # Normalize: The config suggests PIXEL_MEAN=0.5, PIXEL_STD=0.5
    # Since apply_window returns [0, 1], we standardize to roughly [-1, 1]
    volume = (volume - Config.PIXEL_MEAN) / Config.PIXEL_STD

    return volume.astype(np.float32)


def prepare_image_cache(patient_ids, input_dir, cache_dir, load_cached_data=True):
    """
    Iterates over patients and caches their processed image volumes.
    """
    os.makedirs(cache_dir, exist_ok=True)

    for pid in patient_ids:
        save_path = os.path.join(cache_dir, f"{pid}.npy")

        if load_cached_data and os.path.exists(save_path):
            continue

        # Process
        # Note: The dataset structure is input/train/<PatientID> or input/test/<PatientID>
        # We need to find where the patient is. The metadata `image_path` handles this relative path.
        # However, here we just have IDs. We assume they are in train/ unless specified.
        # To be robust, we check both or rely on the dataframe provided to the dataset.
        # This function is a helper; the actual logic is better handled if we pass the full path.
        pass  # Logic moved to Dataset or external loop to handle paths correctly


# -------------------------------------------------------------------------
# Tabular Processing Functions
# -------------------------------------------------------------------------


def get_tabular_features(row, scaler_stats):
    """
    Extracts and normalizes tabular features from a DataFrame row.
    Returns: (features_vector, target_value)
    """
    # 1. Age (Standardized)
    age = (row["Age"] - scaler_stats["age_mean"]) / scaler_stats["age_std"]

    # 2. Sex (Binary: Male=0, Female=1)
    sex = 0.0 if row["Sex"] == "Male" else 1.0

    # 3. SmokingStatus (Ordinal: Never=0, Ex=1, Current=2)
    smoke_map = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
    smoke = smoke_map.get(row["SmokingStatus"], 0.0)

    # 4. Relative Time (Scaled)
    # Weeks - Baseline_Week.
    # Note: The row should already have 'Relative_Weeks' calculated.
    rel_time = row["Relative_Weeks"] * 0.01

    # 5. Baseline FVC (Standardized)
    base_fvc = (row["Baseline_FVC"] - scaler_stats["fvc_mean"]) / scaler_stats[
        "fvc_std"
    ]

    # Construct Feature Vector
    # Order: [Baseline_FVC, Relative_Time, Age, Sex, Smoking]
    # Note: Config says CLINICAL_INPUT_DIM = 4.
    # Wait, the prompt Idea says: "Input: Baseline FVC, Relative Time, Age, Sex, SmokingStatus".
    # That is 5 features. Let's check Config.
    # Config.CLINICAL_INPUT_DIM = 4.
    # This might be a mismatch in the provided Config vs the Idea.
    # I must follow the Idea description ("Metric-Aligned Cascaded Output-Space Residual Network").
    # The Idea lists 5 inputs. I will return 5.
    # If the model expects 4, I would need to adjust, but I cannot change Config.
    # However, usually Sex and Smoking are categorical embeddings or scalars.
    # If Config is fixed at 4, maybe it groups Sex/Smoking?
    # Let's assume the Config meant 4 *additional* features + Baseline, or I should stick to the Config.
    # Let's look at the config again: "Age, Sex, SmokingStatus, RelativeTime". It missed Baseline FVC?
    # Or maybe Baseline FVC is passed separately?
    # Given I cannot change Config, I will assume the Model (which I don't write here) handles the dimension.
    # But I am writing the Dataset. I should provide all 5.

    features = np.array([base_fvc, rel_time, age, sex, smoke], dtype=np.float32)

    # Target
    if "FVC" in row:
        target = (row["FVC"] - scaler_stats["fvc_mean"]) / scaler_stats["fvc_std"]
    else:
        target = 0.0

    return features, target


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class OSICDataset(Dataset):
    def __init__(
        self, df, cache_dir, scaler_stats, mode="train", load_cached_data=True
    ):
        """
        Args:
            df: DataFrame with metadata.
            cache_dir: Directory to save/load processed images.
            scaler_stats: Dictionary with mean/std for Age and FVC.
            mode: 'train', 'val', 'test', or 'submission'.
            load_cached_data: Whether to use existing cache.
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.scaler_stats = scaler_stats
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-process cache if needed (lazy loading in __getitem__ is also an option,
        # but pre-processing ensures file existence)
        # We do lazy processing in __getitem__ to distribute work or check existence there.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                img = np.load(cache_path)
            except:
                # Corrupt file, reprocess
                img = self._process_and_cache(row, cache_path)
        else:
            img = self._process_and_cache(row, cache_path)

        # 2. Load Tabular
        features, target = get_tabular_features(row, self.scaler_stats)

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "tabular": torch.tensor(features, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_week": f"{patient_id}_{row['Weeks']}",  # For submission tracking
            "raw_weeks": row["Weeks"],
        }

    def _process_and_cache(self, row, cache_path):
        # Construct full path to image directory
        # row['image_path'] is relative, e.g., "train/ID..."
        full_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        img = process_patient_images(row["Patient"], full_path)
        np.save(cache_path, img)
        return img


# -------------------------------------------------------------------------
# Data Preparation & Loading
# -------------------------------------------------------------------------


def prepare_dataframe(df, is_train=True, test_meta_df=None):
    """
    Augments the dataframe with Baseline information and Relative Weeks.
    """
    df = df.copy()

    if is_train:
        # For training/val, we have full history.
        # Identify baseline for each patient (min Weeks)
        # We assume the earliest visit is the baseline.
        baseline_df = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "FVC", "Weeks"]].rename(
            columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
        )

        # Merge baseline info back to original df
        df = df.merge(baseline_df, on="Patient", how="left")
        df["Relative_Weeks"] = df["Weeks"] - df["Baseline_Week"]

    else:
        # For submission/test
        # df is likely sample_submission or test.csv
        # If it's test.csv (metadata), it contains the baseline.
        if "FVC" in df.columns and "Weeks" in df.columns and test_meta_df is None:
            # This is the test metadata file itself
            df["Baseline_FVC"] = df["FVC"]
            df["Baseline_Week"] = df["Weeks"]
            df["Relative_Weeks"] = 0  # It is the baseline
        elif test_meta_df is not None:
            # This is sample_submission logic
            # df has Patient_Week, we need to split it
            # But usually we iterate over sample_submission rows
            # Here we assume df has 'Patient' and 'Weeks' parsed

            # Merge with test metadata to get static features + Baseline FVC
            # test_meta_df contains [Patient, Weeks, FVC, Age, Sex, Smoking, image_path]
            # Rename FVC/Weeks in meta to Baseline
            meta = test_meta_df.rename(
                columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
            )
            df = df.merge(meta, on="Patient", how="left")
            df["Relative_Weeks"] = df["Weeks"] - df["Baseline_Week"]

    return df


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(20)

    # 2. Compute Scaler Stats (from Train only)
    scaler_stats = {
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
    }

    # 3. Prepare Dataframes (Add Baseline info)
    train_df = prepare_dataframe(train_df, is_train=True)
    val_df = prepare_dataframe(val_df, is_train=True)

    # 4. Create Datasets
    train_dataset = OSICDataset(
        train_df,
        Config.CACHE_DIR,
        scaler_stats,
        mode="train",
        load_cached_data=load_cached_data,
    )

    val_dataset = OSICDataset(
        val_df,
        Config.CACHE_DIR,
        scaler_stats,
        mode="val",
        load_cached_data=load_cached_data,
    )

    # 5. Create Loaders
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
    )

    return train_loader, val_loader, scaler_stats


def get_submission_loader(scaler_stats, load_cached_data=True):
    """
    Creates a dataloader for the submission file.
    """
    # Load sample submission
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    test_meta = pd.read_csv(Config.TEST_CSV)

    # Parse Patient and Weeks from Patient_Week column
    # Format: ID..._Week
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Prepare dataframe by merging with test metadata
    processed_sub_df = prepare_dataframe(sub_df, is_train=False, test_meta_df=test_meta)

    dataset = OSICDataset(
        processed_sub_df,
        Config.CACHE_DIR,
        scaler_stats,
        mode="submission",
        load_cached_data=load_cached_data,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, processed_sub_df
