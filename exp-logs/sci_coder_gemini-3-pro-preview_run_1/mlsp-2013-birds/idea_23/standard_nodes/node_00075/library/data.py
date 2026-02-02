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
    PyTorch Dataset for Bird Species Classification.
    Handles dynamic loading of spectrograms, channel replication, and label extraction.
    """

    def __init__(self, df, transforms=None, train_mode=True, pseudo_labels=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transformations to apply.
            train_mode (bool): Whether the dataset is used for training (returns targets).
            pseudo_labels (pd.DataFrame, optional): DataFrame containing pseudo-labels for test data.
                                                    Should be indexed by 'rec_id'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.train_mode = train_mode
        self.pseudo_labels = pseudo_labels

        # Pre-compute label column names
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # 1. Resolve Image Path
        # Metadata contains relative path to wav: e.g., essential_data/src_wavs/PC10_... .wav
        # Spectrograms are in Config.SPECTROGRAM_DIR with .bmp extension
        wav_rel_path = row["file_path"]
        wav_filename = os.path.basename(wav_rel_path)
        bmp_filename = os.path.splitext(wav_filename)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # 2. Load Image
        # Load as grayscale (0)
        image = cv2.imread(img_path, 0)

        if image is None:
            # Fallback for missing images (should not happen based on EDA, but for safety)
            # Create a black image of expected size if file read fails
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # 3. Input Adaptation: Channel Replication (Grayscale -> RGB)
        # Stack the single channel 3 times
        image = np.stack([image, image, image], axis=-1)

        # 4. Apply Augmentations/Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 5. Get Labels
        if self.train_mode:
            # Check if we have specific pseudo-labels for this sample (Student Training)
            if self.pseudo_labels is not None and rec_id in self.pseudo_labels.index:
                # Assuming pseudo_labels df has columns 0..18 or similar that match label_cols indices
                # We expect pseudo_labels to be a DataFrame with rec_id index and columns for probs
                target = self.pseudo_labels.loc[rec_id].values.astype(np.float32)
            else:
                # Ground truth from metadata
                target = row[self.label_cols].values.astype(np.float32)

            return image, torch.tensor(target)
        else:
            # Inference mode, return image and rec_id for submission mapping
            return image, torch.tensor(rec_id)


def get_transforms(data="train"):
    """
    Returns Albumentations transforms for train or validation/test.

    Args:
        data (str): 'train' or 'valid'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout (CoarseDropout)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_HEIGHT * 0.2),
                    max_width=int(Config.IMG_WIDTH * 0.2),
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Input-level Mixup to the batch.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup beta distribution parameter.
        device (str): Device to move indices to.

    Returns:
        mixed_x, y_a, y_b, lam
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


def get_train_dataloader(fold=0, use_pseudo_labels=False):
    """
    Creates the training DataLoader.

    Args:
        fold (int): The fold to use for validation (rest is training).
                    Note: Metadata is already split into train.csv (Fold 0) and val.csv.
                    This argument is kept for compatibility but logic uses fixed CSVs.
        use_pseudo_labels (bool): If True, attempts to load pseudo-labels and combine with training data.

    Returns:
        DataLoader
    """
    df_train = pd.read_csv(Config.TRAIN_CSV)

    pseudo_labels_df = None

    if use_pseudo_labels:
        if os.path.exists(Config.PSEUDO_LABEL_PATH):
            print(f"Loading pseudo-labels from {Config.PSEUDO_LABEL_PATH}...")
            # Load pseudo-labels: Expected format parquet with rec_id index and probability columns
            try:
                # We assume the parquet file has columns matching species indices or names
                # For simplicity in this pipeline, we assume the parquet was saved with columns "0", "1", ... "18"
                # and "rec_id" as a column or index.
                pl_df = pd.read_parquet(Config.PSEUDO_LABEL_PATH)

                # Load test metadata to get file paths for these pseudo-labels
                df_test = pd.read_csv(Config.TEST_CSV)

                # Filter test metadata to only include rows we have pseudo-labels for
                # (Though we expect 1:1 mapping)
                if "rec_id" in pl_df.columns:
                    pl_df = pl_df.set_index("rec_id")

                # Ensure columns are sorted 0..18 strings or ints
                pl_cols = [str(i) for i in range(Config.NUM_CLASSES)]
                if not all(c in pl_df.columns for c in pl_cols):
                    # Try int columns
                    pl_cols = [i for i in range(Config.NUM_CLASSES)]

                pseudo_labels_df = pl_df[pl_cols]

                # Filter test df
                df_test_pl = df_test[
                    df_test["rec_id"].isin(pseudo_labels_df.index)
                ].copy()

                # Combine Train + Test(Pseudo)
                # We concatenate the dataframes. The Dataset class handles looking up targets.
                # For df_train, pseudo_labels_df lookup will fail (rec_id not in index), so it uses ground truth.
                # For df_test_pl, pseudo_labels_df lookup succeeds.
                df_combined = pd.concat(
                    [df_train, df_test_pl], axis=0, ignore_index=True
                )

                print(
                    f"Combined Train ({len(df_train)}) + Pseudo-Labeled Test ({len(df_test_pl)}) = {len(df_combined)} samples."
                )
                df_train = df_combined

            except Exception as e:
                print(
                    f"Failed to load pseudo-labels: {e}. Proceeding with labeled data only."
                )
        else:
            print(
                f"Pseudo-label file not found at {Config.PSEUDO_LABEL_PATH}. Proceeding with labeled data only."
            )

    dataset = BirdDataset(
        df=df_train,
        transforms=get_transforms("train"),
        train_mode=True,
        pseudo_labels=pseudo_labels_df,
    )

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )


def get_val_dataloader():
    """
    Creates the validation DataLoader.
    """
    df_val = pd.read_csv(Config.VAL_CSV)

    dataset = BirdDataset(
        df=df_val, transforms=get_transforms("valid"), train_mode=True
    )

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )


def get_test_dataloader():
    """
    Creates the test DataLoader for inference.
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    dataset = BirdDataset(
        df=df_test, transforms=get_transforms("valid"), train_mode=False
    )

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )
