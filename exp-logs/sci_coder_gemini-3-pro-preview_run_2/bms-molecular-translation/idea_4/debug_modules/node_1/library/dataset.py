import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import Tokenizer


def load_metadata(csv_path, cache_filename, load_cached_data=True):
    """
    Loads metadata from CSV, with caching to Parquet for faster subsequent loads.
    Strictly follows the required caching logic.
    """
    # Ensure cache directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached metadata from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. IF loading fails OR load_cached_data is False: Compute/process from scratch.
    print(f"Loading metadata from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source metadata file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    print(f"Caching metadata to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    # 3. Return the data.
    return df


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for the dataset.
    """
    # Normalize to [0, 1] by dividing by 255 (default behavior of A.Normalize with mean=0, std=1)
    # We maintain 1 channel as per Config.
    return A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(mean=(0,), std=(1,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


class InChiDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            tokenizer (Tokenizer): Instance of library.tokenizer.Tokenizer.
            transform (albumentations.Compose): Transforms to apply to images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative to input_dir (e.g., "train/0/0/0/id.png")
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Load as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for corrupt images (should not happen in this dataset based on EDA)
            # Create a black image of correct size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

        # Expand dims to (H, W, 1) for albumentations if it expects channel dim
        # But grayscale loaded by cv2 is (H, W). Albumentations handles this but explicit is better.
        image = image[:, :, np.newaxis]

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W)

        # 3. Handle Label / Mode
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            # Convert text to sequence indices
            label_tensor = self.tokenizer.text_to_sequence(
                inchi_text, max_length=Config.MAX_SEQUENCE_LENGTH, padding=True
            )
            # Calculate actual length (excluding padding) for masking/packing if needed later
            # We can deduce this from the tensor, but passing it explicitly is helpful
            # Note: text_to_sequence adds SOS and EOS.
            # We count non-padding elements.
            seq_len = (
                (label_tensor != self.tokenizer.stoi[Config.PAD_TOKEN]).sum().item()
            )

            return image, label_tensor, seq_len

        else:
            # Test mode: return image and image_id for submission
            image_id = row["image_id"]
            return image, image_id


def get_train_dataloader(
    tokenizer, batch_size=None, debug=False, load_cached_data=True
):
    """
    Creates the training dataloader.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    df = load_metadata(
        Config.TRAIN_METADATA_PATH,
        "train_metadata.parquet",
        load_cached_data=load_cached_data,
    )

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
        print(f"Debug mode: Training data subset to {len(df)} samples.")

    dataset = InChiDataset(
        df, tokenizer, transform=get_transforms(mode="train"), mode="train"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def get_val_dataloader(tokenizer, batch_size=None, debug=False, load_cached_data=True):
    """
    Creates the validation dataloader.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    df = load_metadata(
        Config.VAL_METADATA_PATH,
        "val_metadata.parquet",
        load_cached_data=load_cached_data,
    )

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
        print(f"Debug mode: Validation data subset to {len(df)} samples.")

    dataset = InChiDataset(
        df, tokenizer, transform=get_transforms(mode="val"), mode="val"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def get_test_dataloader(tokenizer, batch_size=None, debug=False, load_cached_data=True):
    """
    Creates the test dataloader.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    df = load_metadata(
        Config.TEST_METADATA_PATH,
        "test_metadata.parquet",
        load_cached_data=load_cached_data,
    )

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
        print(f"Debug mode: Test data subset to {len(df)} samples.")

    dataset = InChiDataset(
        df, tokenizer, transform=get_transforms(mode="test"), mode="test"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader
