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
    from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS,
                    contrast_limit=Config.AUG_CONTRAST,
                    p=0.5,
                ),
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


def load_and_cache_images(df, cache_dir, load_cached_data=True):
    """
    Loads spectrogram images into memory, using a cache file if available.
    Strictly follows the caching logic requirement.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "spectrograms.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    data_dict = {}
    unique_files = df["file_path"].unique()

    for rel_path in unique_files:
        # Map wav path to bmp path
        # rel_path example: essential_data/src_wavs/PC10_... .wav
        basename = os.path.basename(rel_path)
        bmp_name = basename.replace(".wav", ".bmp")
        full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)

        img = None
        if os.path.exists(full_path):
            # Load as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Create placeholder image if missing/corrupt
            # Using 256x1000 as placeholder based on typical dimensions
            img = np.zeros((256, 1000), dtype=np.uint8)

        data_dict[rel_path] = img

    # 3. Save to cache
    try:
        np.save(cache_path, data_dict)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return data_dict


class BirdDataset(Dataset):
    def __init__(self, df, image_dict, transform=None, mode="train"):
        self.df = df
        self.image_dict = image_dict
        self.transform = transform
        self.mode = mode
        self.tile_size = Config.IMAGE_SIZE
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Retrieve image
        img = self.image_dict.get(file_path)
        if img is None:
            img = np.zeros((256, 1000), dtype=np.uint8)

        h, w = img.shape

        # Time Shifting (Train only)
        # Roll along width (time axis)
        if self.mode == "train":
            shift = np.random.randint(0, w)
            img = np.roll(img, shift, axis=1)

        # Tiling Strategy
        # We extract 3 square crops: Start, Middle, End.
        # Crop size is defined by height (H).
        crop_size = h

        # Handle case where width < height (unlikely)
        if w < crop_size:
            pad = crop_size - w
            img = np.pad(img, ((0, 0), (0, pad)), mode="constant")
            w = crop_size

        # Coordinates
        x_start = 0
        x_end = w - crop_size
        x_mid = (w - crop_size) // 2

        starts = [x_start, x_mid, x_end]
        tiles = []

        for x in starts:
            # Crop
            tile = img[0:h, x : x + crop_size]

            # Resize to 224x224
            if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                tile = cv2.resize(
                    tile,
                    (self.tile_size, self.tile_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Replicate to 3 Channels (Grayscale -> RGB)
            tile_rgb = cv2.merge([tile, tile, tile])

            # Apply Transforms
            if self.transform:
                res = self.transform(image=tile_rgb)
                tile_t = res["image"]
            else:
                # Fallback
                tile_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0

            tiles.append(tile_t)

        # Stack tiles: (3, 3, 224, 224)
        image_tensor = torch.stack(tiles, dim=0)

        # Labels
        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        if self.mode != "test":
            label_str = str(row["labels"])
            if label_str != "?" and label_str != "nan":
                parts = label_str.split()
                for p in parts:
                    try:
                        label_vec[int(p)] = 1.0
                    except:
                        pass

        return image_tensor, torch.tensor(label_vec)


def make_folds(train_csv, val_csv, num_folds=5, seed=42):
    """
    Combines train and val data, then creates stratified folds.
    """
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # Combine
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Prepare X (indices) and y (binary matrix) for stratification
    X = df.index.values.reshape(-1, 1)

    y = np.zeros((len(df), Config.NUM_CLASSES))
    for idx, row in df.iterrows():
        l_str = str(row["labels"])
        if l_str != "?" and l_str != "nan":
            for x in l_str.split():
                y[idx, int(x)] = 1

    # Stratification
    if HAS_SKMULTILEARN:
        ms = IterativeStratification(n_splits=num_folds, order=1)
        splits = list(ms.split(X, y))
    else:
        # Fallback to KFold
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(X, y))

    df["fold"] = -1
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = fold_i

    return df


def get_dataloaders(fold_idx, load_cached_data=True, debug=False):
    """
    Constructs DataLoaders for the given fold.
    """
    seed_everything(Config.SEED)

    # 1. Create Folds
    df_folds = make_folds(
        Config.TRAIN_CSV, Config.VAL_CSV, num_folds=Config.NUM_FOLDS, seed=Config.SEED
    )
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Debug Subsampling
    if debug:
        df_folds = df_folds.sample(
            n=min(len(df_folds), Config.DEBUG_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SAMPLES), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Split
    train_df = df_folds[df_folds["fold"] != fold_idx].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold_idx].reset_index(drop=True)

    # 4. Load Images (Cache)
    all_files = pd.concat([df_folds, df_test], ignore_index=True)
    image_dict = load_and_cache_images(
        all_files, Config.CACHE_DIR, load_cached_data=load_cached_data
    )

    # 5. Create Datasets
    train_ds = BirdDataset(
        train_df, image_dict, transform=get_transforms("train"), mode="train"
    )
    val_ds = BirdDataset(
        val_df, image_dict, transform=get_transforms("val"), mode="val"
    )
    test_ds = BirdDataset(
        df_test, image_dict, transform=get_transforms("test"), mode="test"
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
