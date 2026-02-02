import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything


def get_img_path(file_path, data_source):
    """
    Resolves the image path based on the metadata file_path and data_source.
    metadata file_path: essential_data/src_wavs/filename.wav
    target: [spectrogram_dir]/filename.bmp
    """
    filename = os.path.basename(file_path)
    # Replace extension .wav with .bmp
    filename = os.path.splitext(filename)[0] + ".bmp"

    if data_source == "standard":
        return os.path.join(Config.SPECTROGRAM_DIR, filename)
    elif data_source == "filtered":
        return os.path.join(Config.FILTERED_SPECTROGRAM_DIR, filename)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


class TimeShiftPadding(A.ImageOnlyTransform):
    """
    Applies horizontal translation using Zero-Padding (not wrapping).
    Preserves temporal causality by not wrapping the end of the clip to the beginning.
    """

    def __init__(self, limit=0.2, always_apply=False, p=0.5):
        super(TimeShiftPadding, self).__init__(always_apply, p)
        self.limit = limit

    def apply(self, img, **params):
        h, w, c = img.shape
        shift_amt = int(w * self.limit)
        # Random integer between -shift_amt and shift_amt
        shift = np.random.randint(-shift_amt, shift_amt + 1)

        if shift == 0:
            return img

        new_img = np.zeros_like(img)
        if shift > 0:
            # Shift right: move content to right, pad left with zeros
            new_img[:, shift:, :] = img[:, :-shift, :]
        else:
            # Shift left: move content to left, pad right with zeros
            # shift is negative here, so slicing logic uses it directly
            new_img[:, :shift, :] = img[:, -shift:, :]

        return new_img

    def get_transform_init_args_names(self):
        return ("limit",)


def get_transforms(phase):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                # Horizontal Translation (Zero-Padding)
                TimeShiftPadding(limit=0.2, p=0.5),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    def __init__(self, df, data_source="standard", phase="train", transform=None):
        """
        Args:
            df: DataFrame containing metadata (rec_id, file_path, labels).
            data_source: 'standard' or 'filtered'.
            phase: 'train', 'val', or 'test'.
            transform: Albumentations transform pipeline.
        """
        self.df = df
        self.data_source = data_source
        self.phase = phase
        self.transform = transform
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Resolve Image Path
        img_path = get_img_path(file_path, self.data_source)

        # Load Image (Grayscale)
        # Note: Spectrograms are stored as BMPs. We load as grayscale then replicate channels.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for safety, though metadata validation should prevent this
            img = np.zeros(Config.IMG_SIZE, dtype=np.uint8)

        # Resize to fixed dimensions (224x224)
        img = cv2.resize(img, Config.IMG_SIZE)

        # 3-Channel Rule: Replicate grayscale to RGB
        img = cv2.merge([img, img, img])

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # Fallback manual conversion if no transform
            img = img.astype(np.float32) / 255.0
            img = torch.tensor(img).permute(2, 0, 1)

        # Process Labels
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        if self.phase != "test":
            labels_str = str(row["labels"])
            # Parse space-separated indices
            if labels_str != "?" and labels_str.lower() != "nan" and labels_str.strip():
                try:
                    indices = [int(x) for x in labels_str.split()]
                    # Clamp indices just in case
                    indices = [i for i in indices if 0 <= i < self.num_classes]
                    label_vec[indices] = 1.0
                except ValueError:
                    pass

        return img, label_vec


def mixup_data(x, y, alpha=0.4, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_data_splits(load_cached_data=True):
    """
    Merges Train and Validation metadata, performs Iterative Stratification to create 5 folds,
    and returns the dataframe with a 'fold' column.
    Caches the result to parquet.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Merge to maximize data for CV
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Prepare X (ids) and y (labels) for stratification
    X = df["rec_id"].values.reshape(-1, 1)

    # Parse labels to binary matrix for stratification
    num_classes = Config.NUM_CLASSES
    y = np.zeros((len(df), num_classes))

    for idx, row in df.iterrows():
        l_str = str(row["labels"])
        if l_str != "?" and l_str.lower() != "nan" and l_str.strip():
            try:
                indices = [int(x) for x in l_str.split()]
                indices = [i for i in indices if 0 <= i < num_classes]
                y[idx, indices] = 1
            except ValueError:
                pass

    # Iterative Stratification
    # Using order=1 to balance label combinations
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    df["fold"] = -1

    # Assign folds
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df.iloc[val_indices, df.columns.get_loc("fold")] = fold_idx

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path)

    return df
