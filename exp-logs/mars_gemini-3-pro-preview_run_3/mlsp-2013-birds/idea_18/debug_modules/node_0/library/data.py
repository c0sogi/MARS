import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything


class TimeShiftPadding:
    """
    Randomly shifts the image horizontally and pads with zeros.
    Does not wrap around. Preserves temporal causality.
    """

    def __init__(self, max_shift_ratio=0.1, p=0.5):
        self.max_shift_ratio = max_shift_ratio
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        # img is a PIL Image
        w, h = img.size
        max_shift = int(w * self.max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift == 0:
            return img

        # Create a black background
        new_img = Image.new("RGB", (w, h), (0, 0, 0))

        # Paste the shifted image
        # If shift > 0 (shift right), paste at (shift, 0)
        # If shift < 0 (shift left), paste at (shift, 0) which crops left
        new_img.paste(img, (shift, 0))

        return new_img


class BirdDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'labels' (or 'rec_id').
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative to input_dir, e.g., essential_data/src_wavs/file.wav
        # We need to map this to the spectrogram path
        wav_filename = os.path.basename(row["file_path"])
        spec_path = Config.get_spectrogram_path(wav_filename)

        try:
            # Load BMP
            image = Image.open(spec_path).convert("L")  # Load as grayscale
        except Exception as e:
            # Fallback for missing files (should not happen based on metadata check)
            # Create a blank image
            image = Image.new("L", Config.IMG_SIZE)

        # 2. Resize to strict 224x224
        image = image.resize(Config.IMG_SIZE, Image.Resampling.BICUBIC)

        # 3. Convert to RGB (Replicate channels)
        # We do this before transforms because some transforms expect RGB
        image = image.convert("RGB")

        # 4. Apply Transforms
        if self.transform:
            image = self.transform(image)

        # 5. Convert to Tensor and Normalize
        # Using standard ImageNet normalization
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        image = to_tensor(image)
        image = normalize(image)

        # 6. Return Data
        if self.mode in ["train", "val"]:
            # Parse labels
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            label_str = str(row["labels"])
            if label_str != "?" and label_str.strip():
                indices = [int(x) for x in label_str.split()]
                for i in indices:
                    if 0 <= i < self.num_classes:
                        label_vec[i] = 1.0

            return image, label_vec
        else:
            # Test mode: return image and rec_id
            rec_id = row["rec_id"]
            return image, rec_id


def get_transforms(mode="train"):
    """
    Returns the composition of transforms for the given mode.
    """
    if mode == "train":
        return transforms.Compose(
            [
                # Horizontal Translation (Time Shift)
                TimeShiftPadding(max_shift_ratio=0.15, p=0.5),
                # Photometric Augmentations
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # No Horizontal Flip allowed
            ]
        )
    else:
        # Validation/Test: No augmentation, just resizing (handled in Dataset)
        return None


def get_folds(load_cached_data=True):
    """
    Loads training and validation metadata, combines them, and creates 5 folds
    using Iterative Stratification. Caches the result.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_folds = pd.read_parquet(cache_path)
            # print(f"Loaded folds from cache: {cache_path}")
            return df_folds
        except Exception:
            pass  # Fallback to recomputing

    # 2. Load and Combine Data
    # We use both train.csv and val.csv from metadata to form the full dev set
    # because we want to perform our own 5-fold CV.
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    df_all = pd.concat([train_meta, val_meta], ignore_index=True)

    # 3. Prepare for Stratification
    X = df_all["rec_id"].values.reshape(-1, 1)

    # Create binary label matrix
    num_samples = len(df_all)
    num_classes = Config.NUM_CLASSES
    y = np.zeros((num_samples, num_classes))

    for idx, row in df_all.iterrows():
        label_str = str(row["labels"])
        if label_str != "?" and label_str.strip():
            indices = [int(x) for x in label_str.split()]
            y[idx, indices] = 1

    # 4. Iterative Stratification
    # We use a fixed seed for the splitter
    stratifier = IterativeStratification(
        n_splits=Config.NUM_FOLDS,
        order=1,
        sample_distribution_per_fold=[1.0 / Config.NUM_FOLDS] * Config.NUM_FOLDS,
    )

    # Assign folds
    df_all["fold"] = -1

    # iter_strat returns indices
    fold_idx = 0
    for _, test_index in stratifier.split(X, y):
        df_all.iloc[test_index, df_all.columns.get_loc("fold")] = fold_idx
        fold_idx += 1

    # 5. Cache Data
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df_all.to_parquet(cache_path)
    # print(f"Created and cached folds at: {cache_path}")

    return df_all


def get_dataloaders(fold_id, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_id (int): The fold index (0 to NUM_FOLDS-1) to use as validation.
        load_cached_data (bool): Whether to use cached fold splits.

    Returns:
        train_loader, val_loader
    """
    # Get folds dataframe
    df = get_folds(load_cached_data=load_cached_data)

    # Split train/val
    train_df = df[df["fold"] != fold_id].copy()
    val_df = df[df["fold"] == fold_id].copy()

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, mode="train", transform=get_transforms(mode="train")
    )

    val_dataset = BirdDataset(val_df, mode="val", transform=get_transforms(mode="val"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Can handle larger batch size for val
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates DataLoader for the test set.
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = BirdDataset(
        df_test, mode="test", transform=get_transforms(mode="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
