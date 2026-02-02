import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False
    from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms_list = [
        # Resize to strict 224x224 as per strategy
        A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
    ]

    if mode == "train":
        transforms_list.extend(
            [
                # Horizontal Translation (Time-shifting) with Zero-Padding
                # translate_percent x: (-0.2, 0.2) means shifting up to 20% left or right
                # cval=0 ensures zero-padding for the new pixels
                A.Affine(
                    translate_percent={"x": (-0.2, 0.2), "y": (0, 0)},
                    scale=None,
                    rotate=None,
                    shear=None,
                    cval=0,
                    mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Mixup is handled in the training loop, not here.
                # NO Horizontal Flip as per strategy.
            ]
        )

    transforms_list.extend(
        [
            # Normalize using ImageNet defaults
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles loading BMP spectrograms and parsing multi-label targets.
    """

    def __init__(self, df, data_dir, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'labels'.
            data_dir (str): Root directory containing the images.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # The metadata 'file_path' is like 'essential_data/src_wavs/filename.wav'
        # We need to map this to the spectrogram path in data_dir
        # data_dir will be e.g. input/supplemental_data/spectrograms
        # The file names in spectrogram dirs correspond to wav filenames but with .bmp extension

        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = wav_filename.replace(".wav", ".bmp")
        img_path = os.path.join(self.data_dir, bmp_filename)

        # Load Image
        # cv2.imread loads in BGR. If the source is grayscale BMP,
        # it automatically replicates channels to 3, satisfying the 3-Channel Rule.
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing files (should not happen based on analysis)
            # Create a blank black image
            image = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Labels
        if self.mode in ["train", "val"]:
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            label_str = str(row["labels"])

            if label_str and label_str != "?" and label_str.lower() != "nan":
                try:
                    indices = [int(x) for x in label_str.split()]
                    for cls_idx in indices:
                        if 0 <= cls_idx < self.num_classes:
                            label_vec[cls_idx] = 1.0
                except ValueError:
                    pass  # Empty or malformed label

            return image, label_vec

        else:
            # Test mode: return image and rec_id for submission
            rec_id = row["rec_id"]
            return image, rec_id


def get_folds(load_cached_data=True):
    """
    Generates or loads 5-fold CV splits using Iterative Stratification.
    Merges original train and val sets to create a full development set.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame with a 'fold' column.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to re-creation

    # Load original metadata
    train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_path = os.path.join(Config.METADATA_DIR, "val.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Merge to create full dev set
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Prepare for Iterative Stratification
    X = df["rec_id"].values.reshape(-1, 1)

    # Create binary label matrix
    y = np.zeros((len(df), Config.NUM_CLASSES))
    for idx, row in df.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str and lbl_str != "?":
            try:
                indices = [int(x) for x in lbl_str.split()]
                y[idx, indices] = 1
            except ValueError:
                pass

    # Assign Folds
    df["fold"] = -1

    if HAS_SKMULTILEARN:
        stratifier = IterativeStratification(n_splits=Config.N_FOLDS, order=1)
        # split returns indices
        for fold_idx, (train_indices, val_indices) in enumerate(stratifier.split(X, y)):
            df.iloc[val_indices, df.columns.get_loc("fold")] = fold_idx
    else:
        # Fallback to KFold if skmultilearn is missing (though it is installed)
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
        for fold_idx, (_, val_indices) in enumerate(kf.split(X)):
            df.iloc[val_indices, df.columns.get_loc("fold")] = fold_idx

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(
    fold_idx, data_source, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Creates Train and Validation DataLoaders for a specific fold and data source.

    Args:
        fold_idx (int): The fold index to use for validation (0-4).
        data_source (str): 'standard' or 'filtered'.
        batch_size (int): Batch size.
        debug (bool): If True, subsets data for debugging.

    Returns:
        train_loader, val_loader
    """
    # Identify data directory
    data_dir = None
    for source in Config.DATA_SOURCES:
        if source["name"] == data_source:
            data_dir = source["path"]
            break

    if data_dir is None:
        raise ValueError(f"Unknown data source: {data_source}")

    # Get Folds
    df = get_folds(load_cached_data=True)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)
        # In debug mode, just split arbitrarily if folds aren't preserved well
        df["fold"] = np.random.randint(0, Config.N_FOLDS, size=len(df))

    # Split
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Datasets
    train_dataset = BirdDataset(
        train_df,
        data_dir=data_dir,
        transform=get_transforms(mode="train"),
        mode="train",
    )

    val_dataset = BirdDataset(
        val_df, data_dir=data_dir, transform=get_transforms(mode="val"), mode="val"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(data_source, batch_size=Config.BATCH_SIZE):
    """
    Creates Test DataLoader.

    Args:
        data_source (str): 'standard' or 'filtered'.
        batch_size (int): Batch size.

    Returns:
        test_loader
    """
    # Identify data directory
    data_dir = None
    for source in Config.DATA_SOURCES:
        if source["name"] == data_source:
            data_dir = source["path"]
            break

    test_path = os.path.join(Config.METADATA_DIR, "test.csv")
    df_test = pd.read_csv(test_path)

    test_dataset = BirdDataset(
        df_test, data_dir=data_dir, transform=get_transforms(mode="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
