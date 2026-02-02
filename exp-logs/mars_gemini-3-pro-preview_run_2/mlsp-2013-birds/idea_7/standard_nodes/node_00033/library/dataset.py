import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed

# Constants
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CACHE_DIR = "./working/idea_7"
FILTERED_SPEC_DIR = "./input/supplemental_data/filtered_spectrograms"


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


def mixup_data(x, y, alpha=0.4):
    """
    Applies Mixup augmentation to a batch of data.
    Returns:
        mixed_x: The mixed inputs.
        y_a: The targets for the first image.
        y_b: The targets for the second image.
        lam: The mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class BirdDataset(Dataset):
    def __init__(self, df, phase="train", transform=None, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file paths, labels).
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = transform
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        self.images = None
        self.labels = None

        # Cache filenames
        cache_img_path = os.path.join(CACHE_DIR, f"images_{phase}.npy")
        cache_lbl_path = os.path.join(CACHE_DIR, f"labels_{phase}.npy")

        data_loaded = False

        if load_cached_data:
            if os.path.exists(cache_img_path) and os.path.exists(cache_lbl_path):
                try:
                    # Check if cache matches current dataframe length (simple validation)
                    cached_imgs = np.load(cache_img_path)
                    cached_lbls = np.load(cache_lbl_path)

                    if len(cached_imgs) == len(self.df):
                        self.images = cached_imgs
                        self.labels = cached_lbls
                        data_loaded = True
                        # print(f"Loaded {phase} data from cache.")
                except Exception:
                    data_loaded = False

        if not data_loaded:
            # Process data from scratch
            self.images = []
            self.labels = self.df[self.label_cols].values.astype(np.float32)

            for idx, row in self.df.iterrows():
                # Construct path to FILTERED spectrogram
                # Metadata might point to 'spectrograms', we redirect to 'filtered_spectrograms'
                orig_rel_path = row["file_path_spec"]
                filename = os.path.basename(orig_rel_path)
                full_path = os.path.join(FILTERED_SPEC_DIR, filename)

                if not os.path.exists(full_path):
                    # Fallback to original path if filtered not found (should not happen based on task desc)
                    full_path = os.path.join("./input", orig_rel_path)

                # Load image
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Create a blank image if load fails
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                else:
                    # Resize to target size
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Convert to Pseudo-RGB (Stack 3 channels)
                img = np.stack([img, img, img], axis=-1)
                self.images.append(img)

            self.images = np.array(self.images, dtype=np.uint8)

            # Save to cache
            np.save(cache_img_path, self.images)
            np.save(cache_lbl_path, self.labels)
            # print(f"Processed and cached {phase} data.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Apply Albumentations Transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Apply SpecAugment (Time/Freq Masking) on Tensor if training
        if self.phase == "train":
            img = self.apply_spec_augment(img)

        return img, torch.tensor(label, dtype=torch.float32)

    def apply_spec_augment(
        self, tensor, num_mask=1, freq_mask_param=20, time_mask_param=20
    ):
        """
        Applies Frequency and Time Masking to the tensor (C, H, W).
        Since this is an image (H=Freq, W=Time), we mask rows and columns.
        """
        _, H, W = tensor.shape

        # Frequency Masking (Rows)
        for _ in range(num_mask):
            f = int(np.random.uniform(0, freq_mask_param))
            if f > 0 and H - f > 0:
                f0 = np.random.randint(0, H - f)
                tensor[:, f0 : f0 + f, :] = 0.0

        # Time Masking (Cols)
        for _ in range(num_mask):
            t = int(np.random.uniform(0, time_mask_param))
            if t > 0 and W - t > 0:
                t0 = np.random.randint(0, W - t)
                tensor[:, :, t0 : t0 + t] = 0.0

        return tensor
