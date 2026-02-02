import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
from library.config import Config


def get_img_seq(patient_id, image_dir):
    """
    Loads all DICOM files for a patient, sorts them by slice location.
    """
    files = glob.glob(os.path.join(image_dir, "*.dcm"))
    if not files:
        return []

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            continue

    # Sort by ImagePositionPatient Z coordinate if available, else InstanceNumber
    if not slices:
        return []

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            # Fallback to filename sorting if metadata is missing
            slices.sort(key=lambda x: f.name)

    return slices


def get_lung_window(ds, img_size):
    """
    Converts DICOM pixel array to HU, applies lung window, and resizes.
    Window: Level -600, Width 1500.
    """
    try:
        image = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", -1024.0)

        image = image * slope + intercept

        # Lung Window
        level = Config.WINDOW_LEVEL
        width = Config.WINDOW_WIDTH
        lower = level - width / 2
        upper = level + width / 2

        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)  # Normalize 0-1

        # Resize
        image = cv2.resize(image, (img_size, img_size))

        return image
    except Exception:
        return np.zeros((img_size, img_size), dtype=np.float32)


def calculate_lung_area(ds):
    """
    Estimates lung area using HU thresholding (-1000 to -400).
    """
    try:
        image = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", -1024.0)
        hu = image * slope + intercept

        # Simple threshold for lung parenchyma
        mask = (hu > -1000) & (hu < -400)
        return mask.sum()
    except Exception:
        return 0


def select_slices(slices):
    """
    Selects 3 slices: Anchor (Max Lung Area) + 2 Boundaries (50% threshold).
    """
    if not slices:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # Calculate areas
    areas = [calculate_lung_area(s) for s in slices]

    if sum(areas) == 0:
        # Fallback: middle slice and neighbors
        mid = len(slices) // 2
        selected_indices = [max(0, mid - 1), mid, min(len(slices) - 1, mid + 1)]
    else:
        max_area = np.max(areas)
        idx_max = np.argmax(areas)

        threshold = 0.5 * max_area
        candidates = [i for i, a in enumerate(areas) if a >= threshold]

        if not candidates:
            candidates = [idx_max]

        idx_top = candidates[0]  # Superior boundary
        idx_bottom = candidates[-1]  # Inferior boundary

        # Select Anchor + Boundaries
        selected_indices = [idx_top, idx_max, idx_bottom]

        # Sort indices to maintain spatial order
        selected_indices.sort()

    # Process selected slices
    processed_images = []
    for idx in selected_indices:
        processed_images.append(get_lung_window(slices[idx], Config.IMG_SIZE))

    # Pad if necessary (though logic above ensures 3 indices)
    while len(processed_images) < Config.NUM_SLICES:
        processed_images.append(processed_images[-1])

    # Take exactly NUM_SLICES
    processed_images = processed_images[: Config.NUM_SLICES]

    return np.array(processed_images)  # (3, H, W)


def process_images_for_df(df, load_cached=True):
    """
    Iterates through dataframe, processes images for each unique patient,
    and caches them.
    """
    patient_ids = df["Patient"].unique()
    image_cache = {}

    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Processing images for {len(patient_ids)} patients...")

    for pid in patient_ids:
        cache_path = os.path.join(Config.CACHE_DIR, f"{pid}.npy")

        if load_cached and os.path.exists(cache_path):
            try:
                img_data = np.load(cache_path)
                image_cache[pid] = img_data
                continue
            except Exception:
                pass  # Reload if corrupt

        # Process from scratch
        # Look up the image path for this patient
        patient_rows = df[df["Patient"] == pid]
        if len(patient_rows) == 0:
            continue

        rel_path = patient_rows["image_path"].iloc[0]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        slices = get_img_seq(pid, full_path)
        img_data = select_slices(slices)  # Shape (3, 260, 260)

        # Save to cache
        np.save(cache_path, img_data)
        image_cache[pid] = img_data

    return image_cache


def prepare_tabular(df, mode="train", global_stats=None):
    """
    Processes tabular features:
    1. Identify Baseline FVC and Week.
    2. Calculate Relative Time (scaled by 0.01).
    3. Encode SmokingStatus.
    4. Normalize FVC (Target) if mode != 'test'.
    """
    df = df.copy()

    # 1. Baseline Info
    # Sort by Patient and Weeks to find the initial visit
    df_sorted = df.sort_values(["Patient", "Weeks"])
    baseline_df = df_sorted.groupby("Patient").first().reset_index()

    # Select relevant baseline columns
    baseline_cols = ["Patient", "FVC", "Weeks"]
    baseline_df = baseline_df[baseline_cols].rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge back
    df = df.merge(baseline_df, on="Patient", how="left")

    # 2. Relative Time (Scaled by 0.01 as per idea)
    df["Relative_Time"] = (df["Weeks"] - df["Baseline_Week"]) * 0.01

    # 3. Ordinal Encoding for SmokingStatus
    smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
    df["SmokingStatus_Code"] = (
        df["SmokingStatus"].map(smoking_map).fillna(0).astype(int)
    )

    # 4. Normalize Target (FVC)
    if global_stats is None and mode == "train":
        mean_fvc = df["FVC"].mean()
        std_fvc = df["FVC"].std()
        global_stats = {"mean": mean_fvc, "std": std_fvc}
        print(f"Global Target Stats computed: Mean={mean_fvc:.4f}, Std={std_fvc:.4f}")

    if global_stats is not None:
        # Create scaled target column
        df["FVC_scaled"] = (df["FVC"] - global_stats["mean"]) / global_stats["std"]

    return df, global_stats


def normalize_features(train_df, val_df, test_df):
    """
    Standardizes numerical input features (Baseline_FVC, Age) based on Train stats.
    Relative_Time is already scaled by 0.01 and is not Z-scored.
    """
    features = ["Baseline_FVC", "Age"]
    stats = {}

    for col in features:
        mean = train_df[col].mean()
        std = train_df[col].std()
        stats[col] = {"mean": mean, "std": std}

        train_df[f"{col}_scaled"] = (train_df[col] - mean) / std
        val_df[f"{col}_scaled"] = (val_df[col] - mean) / std
        test_df[f"{col}_scaled"] = (test_df[col] - mean) / std

    return train_df, val_df, test_df, stats


def prepare_data(load_cached=True):
    """
    Main entry point.
    Loads metadata, processes images (with caching), processes tabular data.
    Returns:
        train_df, val_df, test_df: Processed DataFrames.
        image_cache: Dict {PatientID: np.array (3, 260, 260)}.
        target_stats: Dict {'mean': float, 'std': float} for FVC un-scaling.
    """
    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Subsampling data to {Config.DEBUG_SAMPLE_SIZE} patients.")
        train_patients = train_meta["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        train_meta = train_meta[train_meta["Patient"].isin(train_patients)].reset_index(
            drop=True
        )

        val_patients = val_meta["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        val_meta = val_meta[val_meta["Patient"].isin(val_patients)].reset_index(
            drop=True
        )
        # Keep test as is or subsample if needed, usually test is small enough

    # Process Images (Union of all patients involved)
    all_df = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)
    image_cache = process_images_for_df(all_df, load_cached=load_cached)

    # Process Tabular
    # 1. Basic engineering (Baseline extraction, encoding, target normalization)
    train_df, target_stats = prepare_tabular(train_meta, mode="train")
    val_df, _ = prepare_tabular(val_meta, mode="val", global_stats=target_stats)
    test_df, _ = prepare_tabular(test_meta, mode="test", global_stats=target_stats)

    # 2. Feature Standardization (Inputs)
    train_df, val_df, test_df, feature_stats = normalize_features(
        train_df, val_df, test_df
    )

    # Print info
    print(f"Data Preparation Complete.")
    print(f"Train: {len(train_df)} rows")
    print(f"Val: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")

    return train_df, val_df, test_df, image_cache, target_stats
