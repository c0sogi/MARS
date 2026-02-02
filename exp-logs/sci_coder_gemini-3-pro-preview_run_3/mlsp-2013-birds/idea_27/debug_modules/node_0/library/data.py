import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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

# Ensure reproducibility
seed_everything(Config.SEED)


class SpectrogramDataset(Dataset):
    """
    Dataset for loading and processing bird sound spectrograms.
    Implements resizing to 224x224 and 3-channel replication.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Pre-compute paths and labels to speed up __getitem__
        self.paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            # Map wav file path from metadata to spectrogram BMP path
            # Metadata: essential_data/src_wavs/filename.wav
            # Target: supplemental_data/spectrograms/filename.bmp
            wav_rel_path = row["file_path"]
            filename = os.path.basename(wav_rel_path)
            bmp_filename = filename.replace(".wav", ".bmp")
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)
            self.paths.append(full_path)

            # Process labels
            label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            if self.mode != "test":
                lbl_str = str(row["labels"])
                if lbl_str and lbl_str != "nan" and lbl_str != "?":
                    try:
                        indices = [int(x) for x in lbl_str.split()]
                        label_vec[indices] = 1.0
                    except ValueError:
                        pass
            self.labels.append(label_vec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Load image
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing files: black image
            image = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            # Convert BGR (cv2 default) to RGB
            # If image is grayscale (2D), convert to 3-channel RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms (includes resizing and normalization)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Basic fallback transform
            image = cv2.resize(image, Config.IMG_SIZE)
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label, idx


class HistogramDataset(Dataset):
    """
    Dataset for loading Bag-of-Audio-Words features.
    """

    def __init__(self, df, feature_map, mode="train"):
        self.df = df.reset_index(drop=True)
        self.feature_map = feature_map
        self.mode = mode

        self.rec_ids = self.df["rec_id"].values
        self.labels = []

        for _, row in self.df.iterrows():
            label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            if self.mode != "test":
                lbl_str = str(row["labels"])
                if lbl_str and lbl_str != "nan" and lbl_str != "?":
                    try:
                        indices = [int(x) for x in lbl_str.split()]
                        label_vec[indices] = 1.0
                    except ValueError:
                        pass
            self.labels.append(label_vec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec_id = self.rec_ids[idx]
        # Retrieve features from map, default to zeros if missing
        features = self.feature_map.get(
            rec_id, np.zeros(Config.MLP_INPUT_DIM, dtype=np.float32)
        )

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return features, label, idx


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms based on the phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                # Restricted Horizontal Translation (<10% width)
                A.Affine(
                    translate_percent={
                        "x": (-Config.SHIFT_LIMIT, Config.SHIFT_LIMIT),
                        "y": (0, 0),
                    },
                    p=0.5,
                ),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=Config.BRIGHTNESS_LIMIT,
                    contrast_limit=Config.CONTRAST_LIMIT,
                    p=0.5,
                ),
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_histogram_features(load_cached_data=True):
    """
    Parses the histogram_of_segments.txt file and returns a dictionary mapping rec_id to feature vector.
    Implements caching using parquet.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "histogram_features.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_feats = pd.read_parquet(cache_path)
            feature_map = {
                row["rec_id"]: row["features"] for _, row in df_feats.iterrows()
            }
            return feature_map
        except Exception:
            pass  # Fallback to re-parsing

    # 2. Parse from source file
    feature_map = {}
    data_list = []

    if os.path.exists(Config.HISTOGRAM_FILE):
        with open(Config.HISTOGRAM_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                try:
                    # Format: rec_id, val1, val2, ...
                    rec_id = int(parts[0])
                    feats = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                    feature_map[rec_id] = feats
                    data_list.append({"rec_id": rec_id, "features": feats})
                except ValueError:
                    continue

    # 3. Save to cache
    if data_list:
        df_feats = pd.DataFrame(data_list)
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_feats.to_parquet(cache_path)

    return feature_map


def make_folds(df, n_folds=5, seed=Config.SEED):
    """
    Applies Iterative Stratification to create folds for multi-label data.
    """
    df = df.copy()

    # Prepare X and y for stratification
    X = df["rec_id"].values.reshape(-1, 1)

    # Construct binary label matrix
    y = np.zeros((len(df), Config.NUM_CLASSES))
    for idx, row in df.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str and lbl_str != "nan" and lbl_str != "?":
            try:
                indices = [int(x) for x in lbl_str.split()]
                y[idx, indices] = 1
            except ValueError:
                pass

    df["fold"] = -1

    if HAS_SKMULTILEARN:
        mskf = IterativeStratification(n_splits=n_folds, order=1)
        # mskf.split returns indices
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
            df.iloc[val_idx, df.columns.get_loc("fold")] = fold
    else:
        # Fallback to random KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            df.iloc[val_idx, df.columns.get_loc("fold")] = fold

    return df


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup to input batch x and targets y.
    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes loss for mixup: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_data(load_cached_data=True):
    """
    Loads train/val/test metadata and prepares the full training set with folds.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Combine train and val for cross-validation
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Create folds
    full_train_df = make_folds(full_train_df, n_folds=Config.NUM_FOLDS)

    # Load histogram features
    feature_map = load_histogram_features(load_cached_data=load_cached_data)

    if Config.DEBUG:
        full_train_df = full_train_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    return full_train_df, test_df, feature_map
