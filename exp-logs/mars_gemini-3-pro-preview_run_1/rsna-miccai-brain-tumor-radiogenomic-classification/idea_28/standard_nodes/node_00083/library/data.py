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


class MiddleSliceDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.modalities = ["FLAIR", "T1wCE", "T2w"]
        self.file_lists = {}  # Simple in-memory cache for file lists

    def _get_middle_file(self, rel_path):
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Check in-memory cache
        if full_path not in self.file_lists:
            if os.path.exists(full_path):
                # Sort by instance number
                files = sorted(
                    [f for f in os.listdir(full_path) if f.endswith(".dcm")],
                    key=lambda x: int(x.split("-")[-1].split(".")[0]),
                )
                self.file_lists[full_path] = files
            else:
                self.file_lists[full_path] = []

        files = self.file_lists[full_path]
        if not files:
            return None

        # Return middle file
        return os.path.join(full_path, files[len(files) // 2])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        channels = []

        for mod in self.modalities:
            rel_path = row[f"{mod.lower()}_path"]
            file_path = self._get_middle_file(rel_path)

            if file_path:
                img = read_dicom(file_path)
                # Resize if necessary
                if img.shape != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
                    img = cv2.resize(
                        img,
                        (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    )
            else:
                img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

            channels.append(img)

        # Stack channels: (H, W, 3)
        # Cite solution_lesson_node_00023: Independent normalization is handled in read_dicom
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
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Determine Data Source
    if split == "test":
        df = pd.read_csv(Config.TEST_METADATA_PATH)
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

    if Config.DEBUG:
        df = df.head(Config.DEBUG_DATASET_SIZE)

    # 2. Create Dataset and Loader
    transform = get_transforms(split)
    # No external cache needed for MiddleSliceDataset
    dataset = MiddleSliceDataset(df, transform=transform, is_test=(split == "test"))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return loader
