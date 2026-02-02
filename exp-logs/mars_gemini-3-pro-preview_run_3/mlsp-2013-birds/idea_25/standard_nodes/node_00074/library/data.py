import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import seed_everything


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    Implements the 'Safe-Zone' augmentation protocol.
    """
    if phase == "train":
        return A.Compose(
            [
                # Note: Resize is handled during data loading/caching to 224x224.
                # Safe-Zone: Restricted Horizontal Translation (Time-Shift)
                # Limit x-shift to 10%, y-shift to 0. Use Zero-Padding (cval=0).
                A.Affine(
                    translate_percent={
                        "x": (-Config.SHIFT_LIMIT, Config.SHIFT_LIMIT),
                        "y": (0, 0),
                    },
                    rotate=0,
                    scale=1.0,
                    shear=0,
                    mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    p=0.5,
                ),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization & Tensor Conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles in-memory numpy arrays for high efficiency.
    """

    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Handle labels (return None or dummy if not available, though we use 0 vectors for test)
        label = self.labels[idx]

        return image, torch.tensor(label, dtype=torch.float32)


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, use_cuda=True):
    """
    Applies Mixup regularization to the batch.
    Returns:
        mixed_x: The mixed input tensor.
        y_a: The first set of labels.
        y_b: The second set of labels.
        lam: The mixing coefficient.
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


def _load_and_process_images(df, root_dir, img_size):
    """
    Internal helper to load BMP images, convert to RGB, and resize.
    Maps WAV paths from metadata to BMP paths in spectrogram directory.
    """
    images = []

    for _, row in df.iterrows():
        # Metadata contains relative path to WAV: essential_data/src_wavs/PC10_... .wav
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path)
        bmp_filename = filename.replace(".wav", ".bmp")

        # Construct full path to spectrogram
        full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image
        # cv2.imread loads as BGR by default.
        img = cv2.imread(full_path)

        if img is None:
            # Fallback or error handling; usually files are guaranteed by verification script
            # Create a black image to prevent crash, though this shouldn't happen
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize
            img = cv2.resize(img, img_size)

        images.append(img)

    return np.array(images, dtype=np.uint8)


def _process_labels(df, num_classes):
    """
    Internal helper to parse label strings into Multi-Hot vectors.
    """
    labels = []
    for _, row in df.iterrows():
        label_str = row["labels"]
        vec = np.zeros(num_classes, dtype=np.float32)

        if pd.notna(label_str) and label_str != "?" and str(label_str).strip() != "":
            try:
                indices = [int(x) for x in str(label_str).split()]
                vec[indices] = 1.0
            except ValueError:
                pass

        labels.append(vec)
    return np.array(labels, dtype=np.float32)


def get_data(load_cached_data=True, debug=False):
    """
    Main data loading function.
    Handles Caching: Checks for .npy files in Config.WORKING_DIR.
    If missing, processes raw data and saves cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, limits dataset size for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    seed_everything(Config.SEED)

    # Define Cache Paths
    cache_files = {
        "train_img": os.path.join(Config.WORKING_DIR, "train_images.npy"),
        "train_lbl": os.path.join(Config.WORKING_DIR, "train_labels.npy"),
        "val_img": os.path.join(Config.WORKING_DIR, "val_images.npy"),
        "val_lbl": os.path.join(Config.WORKING_DIR, "val_labels.npy"),
        "test_img": os.path.join(Config.WORKING_DIR, "test_images.npy"),
        "test_lbl": os.path.join(Config.WORKING_DIR, "test_labels.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["train_img"])
        y_train = np.load(cache_files["train_lbl"])
        X_val = np.load(cache_files["val_img"])
        y_val = np.load(cache_files["val_lbl"])
        X_test = np.load(cache_files["test_img"])
        y_test = np.load(cache_files["test_lbl"])
    else:
        print("Processing data from scratch...")
        # Load Metadata
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # Process Images
        X_train = _load_and_process_images(
            df_train, Config.SPECTROGRAM_DIR, Config.IMG_SIZE
        )
        X_val = _load_and_process_images(
            df_val, Config.SPECTROGRAM_DIR, Config.IMG_SIZE
        )
        X_test = _load_and_process_images(
            df_test, Config.SPECTROGRAM_DIR, Config.IMG_SIZE
        )

        # Process Labels
        y_train = _process_labels(df_train, Config.NUM_CLASSES)
        y_val = _process_labels(df_val, Config.NUM_CLASSES)
        y_test = _process_labels(df_test, Config.NUM_CLASSES)  # Will be all zeros

        # Save to Cache
        np.save(cache_files["train_img"], X_train)
        np.save(cache_files["train_lbl"], y_train)
        np.save(cache_files["val_img"], X_val)
        np.save(cache_files["val_lbl"], y_val)
        np.save(cache_files["test_img"], X_test)
        np.save(cache_files["test_lbl"], y_test)
        print(f"Data cached to {Config.WORKING_DIR}")

    # Debug Subset
    if debug:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(X_train))
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
        X_val = X_val[:subset_size]
        y_val = y_val[:subset_size]
        # Keep test intact or slice? Usually debug train loop.

    # Create Datasets
    train_dataset = BirdDataset(X_train, y_train, transforms=get_transforms("train"))

    val_dataset = BirdDataset(X_val, y_val, transforms=get_transforms("val"))

    test_dataset = BirdDataset(X_test, y_test, transforms=get_transforms("test"))

    return train_dataset, val_dataset, test_dataset
