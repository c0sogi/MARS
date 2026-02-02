import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
CACHE_DIR = "./working/idea_18"
NUM_CLASSES = 19
IMG_SIZE = (256, 640)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles dynamic loading of spectrograms, preprocessing, and augmentation.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (rec_id, file_path, labels).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Pre-compute label columns
        self.label_cols = [f"species_{i}" for i in range(NUM_CLASSES)]

        # Verify label columns exist
        missing_cols = [c for c in self.label_cols if c not in self.df.columns]
        if missing_cols:
            # If missing (e.g. pure test metadata without dummy cols), fill with 0
            for c in missing_cols:
                self.df[c] = 0.0

        self.labels = self.df[self.label_cols].values.astype(np.float32)
        self.file_paths = self.df["file_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Parse Path
        # Metadata file_path points to: essential_data/src_wavs/PC...wav
        # Target spectrogram is: supplemental_data/spectrograms/PC...bmp
        wav_rel_path = self.file_paths[idx]
        basename = os.path.basename(wav_rel_path)
        filename_no_ext = os.path.splitext(basename)[0]
        spectrogram_path = os.path.join(SPECTROGRAM_DIR, f"{filename_no_ext}.bmp")

        # 2. Load Image
        # Load as grayscale first
        image = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (should not happen based on EDA, but for safety)
            # Create a black image of expected size
            image = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)

        # 3. Channel Replication (Gray -> RGB)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 4. Augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 5. Get Label
        label = self.labels[idx]

        # Return dictionary or tuple? Standard is tuple (img, label)
        # We also return rec_id for tracking/submission
        rec_id = self.df.iloc[idx]["rec_id"]

        return image, torch.tensor(label), rec_id


def get_transforms(mode="train", img_size=IMG_SIZE):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),  # Time inversion
                # Unstructured Cutout / CoarseDropout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size[0] * 0.15),
                    max_width=int(img_size[1] * 0.15),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


class MixupCollate:
    """
    Collate function that applies Mixup to the batch.
    """

    def __init__(self, alpha=0.2, p=1.0):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (image, label, rec_id)
        """
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        rec_ids = [item[2] for item in batch]

        if np.random.random() < self.p:
            batch_size = images.size(0)

            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Random shuffle index
            index = torch.randperm(batch_size)

            # Mix images
            mixed_images = lam * images + (1 - lam) * images[index, :]

            # Mix labels
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]

            return mixed_images, mixed_labels, torch.tensor(rec_ids)

        return images, labels, torch.tensor(rec_ids)


def load_data(load_cached_data=True, pseudo_label_path=None):
    """
    Loads training, validation, and test dataframes.
    Handles caching and pseudo-label merging.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filenames
    suffix = "_pseudo" if pseudo_label_path else ""
    cache_train = os.path.join(CACHE_DIR, f"train_df{suffix}.parquet")
    cache_val = os.path.join(CACHE_DIR, f"val_df.parquet")
    cache_test = os.path.join(CACHE_DIR, f"test_df.parquet")

    # 1. Try Load Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            # print("Loading data from cache...")
            df_train = pd.read_parquet(cache_train)
            df_val = pd.read_parquet(cache_val)
            df_test = pd.read_parquet(cache_test)
            return df_train, df_val, df_test

    # 2. Load from Metadata
    # print("Loading data from metadata...")
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 3. Handle Pseudo Labels (Student Mode)
    if pseudo_label_path and os.path.exists(pseudo_label_path):
        # print(f"Merging pseudo-labels from {pseudo_label_path}...")

        # Load pseudo labels
        # Assumes parquet format with rec_id and species_X columns
        if pseudo_label_path.endswith(".parquet"):
            df_pseudo = pd.read_parquet(pseudo_label_path)
        else:
            df_pseudo = pd.read_csv(pseudo_label_path)

        # Ensure df_pseudo has the right columns
        label_cols = [f"species_{i}" for i in range(NUM_CLASSES)]

        # Merge pseudo labels into test set
        # We drop the existing zero-filled species columns in df_test and merge the new ones
        df_test_pseudo = df_test.drop(columns=label_cols, errors="ignore")
        df_test_pseudo = pd.merge(
            df_test_pseudo, df_pseudo[["rec_id"] + label_cols], on="rec_id", how="left"
        )

        # Handle any missing matches (shouldn't happen if pseudo labels are complete)
        df_test_pseudo[label_cols] = df_test_pseudo[label_cols].fillna(0.0)

        # Concatenate Train (Hard Labels) + Test (Soft Pseudo Labels)
        # Train labels are 0/1, Pseudo labels are 0..1
        df_train = pd.concat([df_train, df_test_pseudo], axis=0, ignore_index=True)

        # Shuffle the combined dataset
        df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 4. Save to Cache
    # print("Saving data to cache...")
    df_train.to_parquet(cache_train)
    df_val.to_parquet(cache_val)
    df_test.to_parquet(cache_test)

    return df_train, df_val, df_test


def get_dataloaders(
    batch_size=32,
    num_workers=4,
    pseudo_label_path=None,
    load_cached_data=True,
    use_mixup=False,
    mixup_alpha=0.2,
):
    """
    Constructs and returns DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        pseudo_label_path (str, optional): Path to pseudo-labels file. If provided,
                                           train set becomes Train + Pseudo-Test.
        load_cached_data (bool): Whether to use cached dataframes.
        use_mixup (bool): Whether to apply Mixup in the training loader.
        mixup_alpha (float): Alpha parameter for Beta distribution in Mixup.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Dataframes
    df_train, df_val, df_test = load_data(load_cached_data, pseudo_label_path)

    # Create Transforms
    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")  # Same for test

    # Create Datasets
    train_dataset = BirdDataset(df_train, transforms=train_transforms, mode="train")
    val_dataset = BirdDataset(df_val, transforms=val_transforms, mode="val")
    test_dataset = BirdDataset(df_test, transforms=val_transforms, mode="test")

    # Define Collate Function (Mixup or Default)
    train_collate = None
    if use_mixup:
        train_collate = MixupCollate(alpha=mixup_alpha)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate,
        drop_last=True,  # Useful for batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
