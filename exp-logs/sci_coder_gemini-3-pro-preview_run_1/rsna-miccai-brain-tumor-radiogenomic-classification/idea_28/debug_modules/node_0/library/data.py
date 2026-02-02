import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything

# Attempt to import pydicom for robust DICOM reading
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def read_dicom(path):
    """
    Reads a DICOM file and returns a normalized float32 numpy array (H, W).
    Tries pydicom first, then falls back to OpenCV.
    """
    img = None

    # Attempt 1: pydicom
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
        except Exception:
            pass

    # Attempt 2: OpenCV
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Return zero image if read fails
    if img is None:
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Handle channels (if read as RGB)
    if img.ndim == 3:
        img = img[:, :, 0]

    # Normalize to [0, 1] float32
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img


def get_image_plane_number(filename):
    """Extracts the image number from filename 'Image-X.dcm'."""
    try:
        # Remove extension and split by '-'
        name = os.path.splitext(filename)[0]
        return int(name.split("-")[-1])
    except:
        return 0


def get_brain_roi(modality_dir):
    """
    Scans a directory of DICOM files to find the brain ROI.
    Returns:
        files: List of filenames sorted by instance number.
        start_idx: Index of the first slice with tissue.
        end_idx: Index of the last slice with tissue.
    """
    if not os.path.exists(modality_dir):
        return [], 0, 0

    files = [f for f in os.listdir(modality_dir) if f.endswith(".dcm")]
    if not files:
        return [], 0, 0

    # Sort files numerically
    files.sort(key=get_image_plane_number)

    # Identify ROI (slices with max pixel > 0)
    first_idx = -1
    last_idx = -1

    # We scan all files to be accurate about the ROI
    for i, f in enumerate(files):
        path = os.path.join(modality_dir, f)
        img = read_dicom(path)
        if img.max() > 0:
            if first_idx == -1:
                first_idx = i
            last_idx = i

    if first_idx == -1:
        # No tissue found, use entire range
        first_idx = 0
        last_idx = len(files) - 1

    return files, first_idx, last_idx


def generate_roi_cache(df, cache_path):
    """
    Generates ROI metadata for all subjects in df and saves to Parquet.
    """
    print(f"Generating ROI cache at {cache_path}...")
    cache_list = []
    modalities = ["FLAIR", "T1wCE", "T2w"]

    total = len(df)
    for idx, row in df.iterrows():
        bid = row["BraTS21ID"]

        for mod in modalities:
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            files, start, end = get_brain_roi(full_path)

            # Store as a flat record
            cache_list.append(
                {
                    "BraTS21ID": bid,
                    "Modality": mod,
                    "StartIdx": start,
                    "EndIdx": end,
                    "Files": ";".join(files),  # Store list as delimited string
                }
            )

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total} subjects...")

    cache_df = pd.DataFrame(cache_list)
    cache_df.to_parquet(cache_path, index=False)
    print("Cache generation complete.")
    return cache_df


class SIRVDataset(Dataset):
    def __init__(self, df, cache_df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.modalities = ["FLAIR", "T1wCE", "T2w"]
        self.depths = Config.RELATIVE_DEPTHS  # [0.4, 0.5, 0.6]

        # Convert cache dataframe to a nested dictionary for O(1) lookup
        # Structure: { BraTS21ID: { Modality: { 'files': [], 'start': int, 'end': int } } }
        self.cache = {}
        for _, row in cache_df.iterrows():
            bid = row["BraTS21ID"]
            mod = row["Modality"]
            if bid not in self.cache:
                self.cache[bid] = {}

            self.cache[bid][mod] = {
                "files": row["Files"].split(";") if row["Files"] else [],
                "start": row["StartIdx"],
                "end": row["EndIdx"],
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bid = row["BraTS21ID"]

        channels = []

        # Retrieve subject cache
        subj_cache = self.cache.get(bid, {})

        # Loop Order: Depths (Outer) -> Modalities (Inner)
        # This creates [Mod1_D1, Mod2_D1, Mod3_D1, Mod1_D2, ...]
        # Matches model expectation: [Peripheral(40%), Center(50%), Peripheral(60%)]
        for depth_ratio in self.depths:
            for mod in self.modalities:
                img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

                if mod in subj_cache:
                    info = subj_cache[mod]
                    files = info["files"]
                    start = info["start"]
                    end = info["end"]

                    if files:
                        roi_len = end - start + 1
                        # Calculate relative index
                        rel_idx = int(start + (roi_len * depth_ratio))
                        rel_idx = min(max(rel_idx, 0), len(files) - 1)

                        file_name = files[rel_idx]
                        # Reconstruct path
                        rel_path = row[f"{mod.lower()}_path"]
                        full_path = os.path.join(Config.INPUT_DIR, rel_path, file_name)

                        loaded_img = read_dicom(full_path)

                        # Resize strictly to ensure stackability
                        if loaded_img.shape != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
                            loaded_img = cv2.resize(
                                loaded_img,
                                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                                interpolation=cv2.INTER_LINEAR,
                            )

                        img = loaded_img

                channels.append(img)

        # Stack channels: (H, W, 9)
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get Target
        if self.is_test:
            target = 0.5
        else:
            target = row["MGMT_value"]

        return image, torch.tensor(target, dtype=torch.float32)


def get_transforms(split):
    """
    Returns Albumentations transforms.
    Train: Spatial augmentations (Flip, Rotate, Elastic, Grid). No Translation/Scale.
    Val/Test: Resize only.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE), ToTensorV2()])


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    fold_idx=None,
):
    """
    Creates a DataLoader for the requested split.
    Handles caching, cross-validation splitting, and dataset creation.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Determine Data Source and Cache Filename
    if split == "test":
        df = pd.read_csv(Config.TEST_METADATA_PATH)
        cache_file = "roi_cache_test.parquet"
        df_for_cache = df
    else:
        # Load full training data
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)

        # Filter exclusions
        df_train = df_train[~df_train["BraTS21ID"].isin(Config.EXCLUDE_CASES)]
        df_val = df_val[~df_val["BraTS21ID"].isin(Config.EXCLUDE_CASES)]

        df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

        # Determine specific split dataframe
        if fold_idx is not None:
            # Dynamic K-Fold Split
            skf = StratifiedKFold(
                n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
            )
            fold_gen = skf.split(df_full, df_full["MGMT_value"])
            train_idx, val_idx = list(fold_gen)[fold_idx]

            if split == "train":
                df = df_full.iloc[train_idx].reset_index(drop=True)
            else:
                df = df_full.iloc[val_idx].reset_index(drop=True)
        else:
            # Static Split from metadata files
            df = df_train if split == "train" else df_val

        cache_file = "roi_cache_train_val.parquet"
        df_for_cache = df_full  # Cache everything to support all folds

    if Config.DEBUG:
        df = df.head(Config.DEBUG_DATASET_SIZE)
        # If debugging, we still might want full cache or just debug cache.
        # For simplicity, we cache what we use or full if available.
        if split != "test":
            df_for_cache = df_full.head(Config.DEBUG_DATASET_SIZE * 2)  # Approximation

    # 2. Manage Cache
    cache_path = os.path.join(Config.CACHE_DIR, cache_file)
    cache_df = None

    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            print(f"Loaded ROI cache from {cache_path}")
        except Exception:
            print("Cache load failed, regenerating...")

    if cache_df is None:
        cache_df = generate_roi_cache(df_for_cache, cache_path)

    # 3. Create Dataset and Loader
    transform = get_transforms(split)
    dataset = SIRVDataset(df, cache_df, transform=transform, is_test=(split == "test"))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return loader
