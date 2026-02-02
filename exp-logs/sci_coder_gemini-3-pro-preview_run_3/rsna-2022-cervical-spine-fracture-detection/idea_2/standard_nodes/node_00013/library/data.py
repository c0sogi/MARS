import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Attempt to import pydicom, handle missing package gracefully
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def load_dicom_slice(path, img_size=256):
    """
    Reads a DICOM file, applies bone windowing, and resizes.
    """
    if not os.path.exists(path):
        # Return black image if file missing (boundary handling)
        return np.zeros((img_size, img_size), dtype=np.float32)

    img = None

    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            # Apply RescaleSlope and RescaleIntercept if they exist
            slope = getattr(dcm, "RescaleSlope", 1.0)
            intercept = getattr(dcm, "RescaleIntercept", 0.0)
            img = img * slope + intercept
        except Exception:
            # Fallback if pydicom fails to read pixel array
            img = None

    # Fallback to cv2 if pydicom missing or failed (e.g. for simple image formats disguised as dcm)
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img.astype(np.float32)
        except Exception:
            pass

    if img is None:
        return np.zeros((img_size, img_size), dtype=np.float32)

    # Bone Windowing
    # Window Center (Level) = 400, Window Width = 1800
    center = 400
    width = 1800

    lower = center - (width / 2)
    upper = center + (width / 2)

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    # Resize
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    return img


def get_transforms(mode="train", img_size=256):
    """
    Returns Albumentations transforms.
    Cite solution_lesson_node_00012: We use independent augmentation per slice (A.Compose)
    instead of consistent augmentation (A.ReplayCompose) to act as a regularizer.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class CervicalSpineDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        transform=None,
        cache_dir=None,
        load_cached=True,
        seq_len=64,
        img_size=256,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.cache_dir = cache_dir
        self.load_cached = load_cached
        self.seq_len = seq_len
        self.img_size = img_size

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Get labels if available
        # Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall
        label_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        if all(c in row for c in label_cols):
            labels = row[label_cols].values.astype(np.float32)
        else:
            labels = np.zeros(8, dtype=np.float32)

        # Handle Caching
        cache_path = (
            os.path.join(self.cache_dir, f"{uid}.npy") if self.cache_dir else None
        )

        volume = None
        if self.load_cached and cache_path and os.path.exists(cache_path):
            try:
                volume = np.load(
                    cache_path
                )  # Expected shape: (seq_len, img_size, img_size, 3)
            except Exception:
                volume = None  # Corrupt cache, recompute

        if volume is None:
            volume = self.process_volume(uid)
            if self.load_cached and cache_path:
                np.save(cache_path, volume)

        # volume is (seq_len, H, W, 3)

        # Prepare output tensor: (seq_len, 3, H, W)
        # Albumentations expects HWC. ToTensorV2 converts to CHW.

        final_sequence = []

        # Cite solution_lesson_node_00012: Apply independent augmentation per slice.
        # This prevents the model from overfitting to specific artifacts and acts as a regularizer.

        if self.transform:
            for i in range(self.seq_len):
                res = self.transform(image=volume[i])
                final_sequence.append(res["image"])
        else:
            # Fallback if no transform provided
            t = ToTensorV2()
            for i in range(self.seq_len):
                final_sequence.append(t(image=volume[i])["image"])

        # Stack to (seq_len, 3, H, W)
        data_tensor = torch.stack(final_sequence)

        return data_tensor, torch.tensor(labels)

    def process_volume(self, uid):
        """
        Loads DICOMs, selects 64 slices, builds 2.5D stack.
        Returns numpy array of shape (seq_len, img_size, img_size, 3)
        """
        study_path = os.path.join(self.image_dir, uid)

        # List all files
        try:
            files = glob.glob(os.path.join(study_path, "*"))
        except Exception:
            files = []

        if len(files) == 0:
            # Return empty volume
            return np.zeros(
                (self.seq_len, self.img_size, self.img_size, 3), dtype=np.float32
            )

        # Sort files.
        # Strategy: Try to extract integer from filename (e.g. "10.dcm" -> 10).
        # This is faster than reading headers and usually correct for RSNA datasets.
        try:
            files = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))
        except ValueError:
            files = sorted(files)  # Fallback to lexicographical

        num_files = len(files)

        # Uniform sampling indices
        indices = np.linspace(0, num_files - 1, self.seq_len).astype(int)

        volume_stack = []

        for idx in indices:
            # 2.5D Stacking: (z-1, z, z+1)
            # Handle boundaries by clamping
            prev_idx = max(0, idx - 1)
            next_idx = min(num_files - 1, idx + 1)

            # We need to read 3 slices (or reuse if cached in memory, but simpler to read)
            # Optimization: In a loop, prev_idx of current might be idx of previous.
            # But here indices are spaced out, so overlap is rare unless few slices.

            paths = [files[prev_idx], files[idx], files[next_idx]]
            channels = []

            for p in paths:
                img = load_dicom_slice(p, self.img_size)
                channels.append(img)

            # Stack channels -> (H, W, 3)
            img_25d = np.stack(channels, axis=-1)
            volume_stack.append(img_25d)

        return np.array(volume_stack, dtype=np.float32)


def get_dataloaders(
    train_metadata_path, val_metadata_path, image_dir, batch_size, num_workers
):
    """
    Creates train and validation dataloaders.
    """
    # Load Metadata
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)

    # Debug Mode Slicing
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Transforms
    train_transform = get_transforms(mode="train", img_size=Config.IMG_SIZE)
    val_transform = get_transforms(mode="val", img_size=Config.IMG_SIZE)

    # Datasets
    train_dataset = CervicalSpineDataset(
        df=train_df,
        image_dir=image_dir,
        transform=train_transform,
        cache_dir=Config.WORKING_DIR,
        load_cached=True,
        seq_len=Config.SEQ_LEN,
        img_size=Config.IMG_SIZE,
    )

    val_dataset = CervicalSpineDataset(
        df=val_df,
        image_dir=image_dir,
        transform=val_transform,
        cache_dir=Config.WORKING_DIR,
        load_cached=True,
        seq_len=Config.SEQ_LEN,
        img_size=Config.IMG_SIZE,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(test_metadata_path, image_dir, batch_size, num_workers):
    """
    Creates test dataloader.
    """
    test_df = pd.read_csv(test_metadata_path)

    # Transforms (same as val)
    test_transform = get_transforms(mode="val", img_size=Config.IMG_SIZE)

    test_dataset = CervicalSpineDataset(
        df=test_df,
        image_dir=image_dir,
        transform=test_transform,
        cache_dir=Config.WORKING_DIR,
        load_cached=True,
        seq_len=Config.SEQ_LEN,
        img_size=Config.IMG_SIZE,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df
