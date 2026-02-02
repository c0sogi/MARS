import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Try importing IterativeStratification, fallback to KFold if not available
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    from sklearn.model_selection import KFold

    HAS_SKMULTILEARN = False


def load_spectrogram_bmp(path):
    """
    Loads a BMP spectrogram, converts to grayscale float32 [0, 1].
    """
    if not os.path.exists(path):
        # Return a blank image if file is missing (should not happen based on checks)
        return np.zeros((224, 500), dtype=np.float32)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((224, 500), dtype=np.float32)

    img = img.astype(np.float32) / 255.0
    return img


def cache_spectrograms(df, cache_dir, load_cached_data=True):
    """
    Caches spectrograms as individual .npy files.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Identify all unique files in the dataframe
    # Map rec_id to file_path
    # Note: df['file_path'] points to .wav in essential_data/src_wavs
    # We need to map this to supplemental_data/spectrograms/*.bmp

    cached_files = {}

    for idx, row in df.iterrows():
        rec_id = row["rec_id"]
        wav_path = row["file_path"]

        # Construct BMP path
        filename = os.path.basename(wav_path)
        bmp_filename = filename.replace(".wav", ".bmp")
        bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        cache_path = os.path.join(cache_dir, f"{rec_id}.npy")

        if load_cached_data and os.path.exists(cache_path):
            continue  # Already cached
        else:
            # Process and save
            img = load_spectrogram_bmp(bmp_path)
            np.save(cache_path, img)


def get_cached_path(rec_id):
    return os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")


class BirdDataset(Dataset):
    def __init__(self, df, mode="train", transform=True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transform (bool): Whether to apply augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Pre-calculate label vectors
        self.labels = []
        if "labels" in df.columns:
            for lbl_str in df["labels"]:
                vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
                if isinstance(lbl_str, str) and lbl_str != "?" and lbl_str.strip():
                    try:
                        indices = [int(x) for x in lbl_str.split()]
                        vec[indices] = 1.0
                    except ValueError:
                        pass
                self.labels.append(vec)
        else:
            # Dummy labels for test if column missing
            self.labels = [
                np.zeros(Config.NUM_CLASSES, dtype=np.float32) for _ in range(len(df))
            ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Load spectrogram from cache
        cache_path = get_cached_path(rec_id)
        if os.path.exists(cache_path):
            img = np.load(cache_path)
        else:
            # Fallback if cache missing (should be handled by setup)
            wav_path = row["file_path"]
            filename = os.path.basename(wav_path)
            bmp_filename = filename.replace(".wav", ".bmp")
            bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)
            img = load_spectrogram_bmp(bmp_path)

        # Augmentation: Horizontal Roll (Time Shift)
        if self.mode == "train" and self.transform:
            # Roll image horizontally
            shift = np.random.randint(-img.shape[1] // 10, img.shape[1] // 10)
            img = np.roll(img, shift, axis=1)

        # Resize to (224, 224)
        # Cite {solution_lesson_node_00030}: Global Resizing vs. Multi-Instance Tiling
        img_resized = cv2.resize(
            img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )

        # Augmentation: Photometric
        # Cite {solution_lesson_node_00009}: Photometric Augmentations on Spectrograms
        if self.mode == "train" and self.transform:
            # Brightness
            brightness = np.random.uniform(-0.1, 0.1)
            img_resized = img_resized + brightness
            # Contrast
            contrast = np.random.uniform(0.8, 1.2)
            img_resized = img_resized * contrast
            # Clip
            img_resized = np.clip(img_resized, 0.0, 1.0)

        # Convert to Tensor and replicate channels
        # Input is (H, W), Output needed (3, H, W)
        tensor = torch.from_numpy(img_resized).float()
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, 224, 224)

        # Normalize (ImageNet stats)
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        # View as (3, 1, 1) to broadcast over H, W
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tensor = (tensor - mean) / std

        label = torch.from_numpy(self.labels[idx]).float()

        return tensor, label, int(rec_id)


def make_folds(train_csv_path, val_csv_path):
    """
    Merges train and val CSVs and creates 5 stratified folds.
    Returns a DataFrame with a 'kfold' column.
    """
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # Combine
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Prepare X and y for stratification
    X = df["rec_id"].values.reshape(-1, 1)

    # Parse labels to binary matrix
    y = np.zeros((len(df), Config.NUM_CLASSES), dtype=int)
    for i, lbl_str in enumerate(df["labels"]):
        if isinstance(lbl_str, str) and lbl_str.strip():
            try:
                indices = [int(x) for x in lbl_str.split()]
                y[i, indices] = 1
            except ValueError:
                pass

    # Initialize folds
    df["kfold"] = -1

    if HAS_SKMULTILEARN:
        # Iterative Stratification
        mskf = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
            df.loc[val_idx, "kfold"] = fold
    else:
        # Fallback to KFold (random but seeded)
        kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            df.loc[val_idx, "kfold"] = fold

    return df


def get_dataloaders(fold_idx=0, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        fold_idx (int): The fold index to use for validation (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached spectrograms.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Prepare Data and Folds
    df_full = make_folds(Config.TRAIN_METADATA_PATH, Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Cache Data
    # We cache everything found in full train/val and test
    all_data = pd.concat([df_full, df_test], ignore_index=True)
    cache_spectrograms(all_data, Config.CACHE_DIR, load_cached_data=load_cached_data)

    # 3. Split Train/Val
    train_df = df_full[df_full["kfold"] != fold_idx].copy()
    val_df = df_full[df_full["kfold"] == fold_idx].copy()

    # Debugging subsample
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]

    # 4. Create Datasets
    train_dataset = BirdDataset(train_df, mode="train", transform=True)
    val_dataset = BirdDataset(val_df, mode="val", transform=False)
    test_dataset = BirdDataset(df_test, mode="test", transform=False)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
