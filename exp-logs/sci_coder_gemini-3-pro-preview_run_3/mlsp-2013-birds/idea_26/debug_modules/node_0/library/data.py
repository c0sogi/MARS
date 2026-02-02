import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False
    from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything

# ==========================================
# Dataset Classes
# ==========================================


class SpectrogramDataset(Dataset):
    """
    Dataset for loading and processing spectrogram images for the CNN stream.
    Reads BMP files, converts to RGB, and applies resizing/augmentation.
    """

    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.root_dir = Config.SPECTROGRAM_DIR

        # Pre-calculate label vectors for training/validation
        self.labels = []
        if mode != "test":
            for _, row in self.df.iterrows():
                label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
                if pd.notna(row["labels"]) and row["labels"] != "?":
                    try:
                        indices = [int(x) for x in str(row["labels"]).split()]
                        label_vec[indices] = 1.0
                    except ValueError:
                        pass  # Handle empty or malformed labels gracefully
                self.labels.append(label_vec)

        # Map file paths from wav to bmp
        self.image_paths = []
        for _, row in self.df.iterrows():
            # Metadata has 'essential_data/src_wavs/filename.wav'
            # We need 'filename.bmp' in SPECTROGRAM_DIR
            wav_path = row["file_path"]
            filename = os.path.basename(wav_path).replace(".wav", ".bmp")
            self.image_paths.append(os.path.join(self.root_dir, filename))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load Image
        # BMPs are single channel usually. Load as grayscale.
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (should not happen based on metadata check)
            image = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)

        # Resize first to ensure consistency with model input
        image = cv2.resize(image, Config.IMAGE_SIZE)

        # Convert to 3 channels (RGB) by replicating the single channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]
            image = image.float() / 255.0

        if self.mode == "test":
            # Return ID for submission mapping
            return image, torch.tensor(self.df.iloc[idx]["rec_id"], dtype=torch.long)
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.float32)


class HistogramDataset(Dataset):
    """
    Dataset for loading Bag-of-Audio-Words features for the MLP stream.
    """

    def __init__(self, df, feature_map, mode="train"):
        self.df = df
        self.feature_map = feature_map  # dict: rec_id -> np.array
        self.mode = mode

        self.labels = []
        if mode != "test":
            for _, row in self.df.iterrows():
                label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
                if pd.notna(row["labels"]) and row["labels"] != "?":
                    try:
                        indices = [int(x) for x in str(row["labels"]).split()]
                        label_vec[indices] = 1.0
                    except ValueError:
                        pass
                self.labels.append(label_vec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec_id = self.df.iloc[idx]["rec_id"]

        # Retrieve features from the pre-loaded map
        if rec_id in self.feature_map:
            features = self.feature_map[rec_id]
        else:
            # Fallback zero vector if ID not found in histogram file
            features = np.zeros(Config.MLP_INPUT_DIM, dtype=np.float32)

        features = torch.tensor(features, dtype=torch.float32)

        if self.mode == "test":
            return features, torch.tensor(rec_id, dtype=torch.long)
        else:
            label = self.labels[idx]
            return features, torch.tensor(label, dtype=torch.float32)


# ==========================================
# Data Processing & Caching
# ==========================================


def load_histogram_features():
    """
    Parses the histogram_of_segments.txt file.
    Returns a dictionary: rec_id -> feature_vector (np.array)
    """
    feature_map = {}
    if not os.path.exists(Config.HISTOGRAM_FILE):
        print(f"Warning: Histogram file not found at {Config.HISTOGRAM_FILE}")
        return feature_map

    with open(Config.HISTOGRAM_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("rec_id"):
                continue
            parts = line.split(",")
            try:
                rec_id = int(parts[0])
                # The rest are features (should be 100 dims)
                feats = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                feature_map[rec_id] = feats
            except ValueError:
                continue
    return feature_map


def prepare_folds(load_cached_data=True):
    """
    Combines train and val metadata, performs 5-fold Iterative Stratification,
    and caches the result.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating new folds with Iterative Stratification...")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Combine to create the full development set
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare X and y for stratification
    X = df["rec_id"].values.reshape(-1, 1)

    # Create binary label matrix for stratification
    y = np.zeros((len(df), Config.NUM_CLASSES))
    for idx, row in df.iterrows():
        if pd.notna(row["labels"]) and row["labels"] != "?":
            indices = [int(x) for x in str(row["labels"]).split()]
            y[idx, indices] = 1

    # Initialize folds
    df["fold"] = -1

    if HAS_SKMULTILEARN:
        # Iterative Stratification for Multi-Label data
        k_fold = IterativeStratification(
            n_splits=Config.NUM_FOLDS, order=1, random_state=Config.SEED
        )
        for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
            df.loc[val_idx, "fold"] = fold
    else:
        # Fallback: StratifiedKFold on string representation of label sets
        print("skmultilearn not found, using StratifiedKFold on label combinations.")
        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        y_str = df["labels"].astype(str)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_str)):
            df.loc[val_idx, "fold"] = fold

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path)

    return df


# ==========================================
# Augmentation
# ==========================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the CNN stream.
    """
    if mode == "train":
        return A.Compose(
            [
                # Resize first
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                # 1. Restricted Horizontal Translation (Time-shift)
                # shift_limit_x=0.1 means +/- 10% of width
                # Uses Zero-Padding (BORDER_CONSTANT, value=0)
                A.ShiftScaleRotate(
                    shift_limit_x=Config.SHIFT_LIMIT,
                    shift_limit_y=0.0,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # 2. Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=Config.BRIGHTNESS_LIMIT,
                    contrast_limit=Config.CONTRAST_LIMIT,
                    p=0.5,
                ),
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


class MixupCollate:
    """
    Collate function that applies Mixup to batches.
    Handles both image batches (CNN) and feature batches (MLP).
    """

    def __init__(self, alpha=Config.MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Args:
            batch: list of (input, target) tuples
        """
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])

        # Only apply mixup if alpha > 0 and we have more than 1 sample
        if self.alpha > 0 and inputs.size(0) > 1:
            lam = np.random.beta(self.alpha, self.alpha)

            batch_size = inputs.size(0)
            index = torch.randperm(batch_size)

            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
            mixed_targets = lam * targets + (1 - lam) * targets[index, :]

            return mixed_inputs, mixed_targets

        return inputs, targets


# ==========================================
# Data Loaders Factory
# ==========================================


def get_dataloaders(
    fold, df_folds, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates dataloaders for both CNN and MLP streams for a specific fold.

    Returns:
        dataloaders (dict): {
            'cnn': {'train': loader, 'val': loader},
            'mlp': {'train': loader, 'val': loader}
        }
    """
    # Split data based on fold
    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    # Load Histogram Features for MLP
    feature_map = load_histogram_features()

    # --- CNN Datasets ---
    train_cnn_ds = SpectrogramDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )
    val_cnn_ds = SpectrogramDataset(val_df, transform=get_transforms("val"), mode="val")

    # --- MLP Datasets ---
    train_mlp_ds = HistogramDataset(train_df, feature_map, mode="train")
    val_mlp_ds = HistogramDataset(val_df, feature_map, mode="val")

    # --- Loaders ---
    # Use MixupCollate for training
    mixup_collate = MixupCollate(alpha=Config.MIXUP_ALPHA)

    train_cnn_loader = DataLoader(
        train_cnn_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mixup_collate,
        pin_memory=True,
        drop_last=True,  # Drop last to ensure mixup stability
    )

    val_cnn_loader = DataLoader(
        val_cnn_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_mlp_loader = DataLoader(
        train_mlp_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mixup_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_mlp_loader = DataLoader(
        val_mlp_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "cnn": {"train": train_cnn_loader, "val": val_cnn_loader},
        "mlp": {"train": train_mlp_loader, "val": val_mlp_loader},
    }


def get_test_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates dataloaders for inference on the test set.
    """
    test_df = pd.read_csv(Config.TEST_CSV)
    feature_map = load_histogram_features()

    # CNN Test Dataset
    test_cnn_ds = SpectrogramDataset(
        test_df, transform=get_transforms("test"), mode="test"
    )
    test_cnn_loader = DataLoader(
        test_cnn_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # MLP Test Dataset
    test_mlp_ds = HistogramDataset(test_df, feature_map, mode="test")
    test_mlp_loader = DataLoader(
        test_mlp_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "cnn": test_cnn_loader,
        "mlp": test_mlp_loader,
        "ids": test_df["rec_id"].values,
    }
