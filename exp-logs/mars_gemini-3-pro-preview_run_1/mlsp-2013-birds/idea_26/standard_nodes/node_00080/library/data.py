import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Loads BMP spectrograms, applies preprocessing and augmentations.
    """

    def __init__(self, df, config, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            config (Config): Configuration object.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.config = config
        self.mode = mode
        self.transform = transform

        # Identify label columns
        # Explicitly select columns based on config to avoid artifacts (Cite debug_lesson_1)
        num_classes = self.config.MODEL_PARAMS["num_classes"]
        self.label_cols = [f"species_{i}" for i in range(num_classes)]

        # Pre-compute file paths to avoid overhead in __getitem__
        self.file_paths = []
        self.labels = []

        for idx, row in self.df.iterrows():
            # Construct spectrogram path from wav file_path in metadata
            # Metadata file_path example: essential_data/src_wavs/PC10_... .wav
            wav_rel_path = row["file_path"]
            wav_basename = os.path.basename(wav_rel_path)
            bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"

            # Spectrograms are in config.SPECTROGRAM_DIR
            img_path = os.path.join(self.config.SPECTROGRAM_DIR, bmp_basename)
            self.file_paths.append(img_path)

            # Get labels
            # For test mode (without pseudo labels), these are 0s.
            # For train/val or student training (with pseudo labels), these are targets.
            lbl = row[self.label_cols].values.astype(np.float32)
            self.labels.append(lbl)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # 1. Load Image
        # Load as grayscale (single channel)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for missing files (should not happen if verified)
            # Create a blank image of expected size
            img = np.zeros(
                (self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.uint8
            )

        # 2. Channel Replication (1 -> 3)
        # Replicate single channel to R, G, B
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 3. Apply Transforms (Resize, Augmentations, Normalize, ToTensor)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # Minimal transform if none provided (Resize + Norm + ToTensor)
            resize = A.Resize(
                height=self.config.IMG_HEIGHT, width=self.config.IMG_WIDTH
            )
            norm = A.Normalize(mean=self.config.MEAN, std=self.config.STD)
            to_tensor = ToTensorV2()

            img = resize(image=img)["image"]
            img = norm(image=img)["image"]
            img = to_tensor(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms(config, mode="train"):
    """
    Returns Albumentations transforms based on mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # High-Fidelity Resolution
                A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
                # Augmentations
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout (CoarseDropout)
                # Random rectangular masks.
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalization
                A.Normalize(mean=config.MEAN, std=config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
                A.Normalize(mean=config.MEAN, std=config.STD),
                ToTensorV2(),
            ]
        )


def get_dataloaders(config, pseudo_labels=None):
    """
    Creates DataLoaders for train, val, and test.

    Args:
        config (Config): Configuration object.
        pseudo_labels (pd.DataFrame, optional): DataFrame containing pseudo-labels for the test set.
                                                Format: [rec_id, species_0, ..., species_18].
                                                If provided, creates a 'student' training set
                                                (Train + Pseudo-Test).

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA)
    val_df = pd.read_csv(config.VAL_METADATA)
    test_df = pd.read_csv(config.TEST_METADATA)

    # Debug mode: reduce dataset size
    if config.DEBUG:
        train_df = train_df.head(10)
        val_df = val_df.head(10)
        test_df = test_df.head(10)

    # Handle Pseudo-labels for Student Training
    if pseudo_labels is not None:
        # Ensure pseudo_labels has the same columns as expected
        print("Merging pseudo-labels into training set...")

        # Drop the placeholder species columns from test_df
        label_cols = [c for c in test_df.columns if c.startswith("species_")]
        test_df_clean = test_df.drop(columns=label_cols)

        # Merge on rec_id
        if "rec_id" in pseudo_labels.columns:
            test_df_pseudo = pd.merge(
                test_df_clean, pseudo_labels, on="rec_id", how="inner"
            )
        else:
            # If pseudo_labels doesn't have rec_id, assume alignment (fallback)
            test_df_pseudo = test_df_clean.copy()
            for col in label_cols:
                if col in pseudo_labels.columns:
                    test_df_pseudo[col] = pseudo_labels[col].values

        # Combine Train and Pseudo-Test
        train_df = pd.concat([train_df, test_df_pseudo], axis=0, ignore_index=True)

        print(
            f"Created Student Dataset: {len(train_df)} samples (Original Train + Pseudo-Labeled Test)"
        )

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, config, mode="train", transform=get_transforms(config, mode="train")
    )

    val_dataset = BirdDataset(
        val_df, config, mode="val", transform=get_transforms(config, mode="val")
    )

    test_dataset = BirdDataset(
        test_df, config, mode="test", transform=get_transforms(config, mode="test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


class Mixup:
    """
    Implements Input-Level Mixup.
    Can be used inside the training loop.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        # Use numpy's default_rng for better random generation, seeded externally via set_seed
        self.rng = np.random.default_rng()

    def __call__(self, x, y):
        """
        Args:
            x (torch.Tensor): Input batch (B, C, H, W).
            y (torch.Tensor): Label batch (B, NumClasses).

        Returns:
            mixed_x, mixed_y
        """
        if self.alpha <= 0:
            return x, y

        batch_size = x.size(0)
        # Sample lambda from beta distribution
        lam = self.rng.beta(self.alpha, self.alpha)

        # Random permutation for mixing
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]

        return mixed_x, mixed_y
