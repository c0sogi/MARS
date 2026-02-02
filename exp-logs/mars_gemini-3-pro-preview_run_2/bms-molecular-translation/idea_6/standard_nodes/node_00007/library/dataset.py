import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.tokenizer import Tokenizer


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI chemical structure recognition.
    Handles image loading, resizing, padding, and label encoding.
    """

    def __init__(
        self,
        df,
        tokenizer,
        image_height=Config.IMAGE_HEIGHT,
        max_width=2048,
        mode="train",
        transform=None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and optionally 'InChI'.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            image_height (int): Target height for resizing.
            max_width (int): Fixed width to pad images to.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations or other transforms.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.image_height = image_height
        self.max_width = max_width
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # 1. Load Image
        # Load as grayscale (1 channel)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for missing images (should ideally not happen given validation)
            # Create a blank white image
            image = np.full((self.image_height, self.image_height), 255, dtype=np.uint8)

        # 2. Preprocess Image
        h, w = image.shape

        # Resize to fixed height, maintaining aspect ratio
        scale = self.image_height / h
        new_w = int(w * scale)

        # Cap width at max_width to prevent OOM on extreme outliers
        if new_w > self.max_width:
            new_w = self.max_width

        image = cv2.resize(image, (new_w, self.image_height))

        # Pad to fixed max_width
        # Create a white canvas (255)
        canvas = np.full((self.image_height, self.max_width), 255, dtype=np.uint8)
        # Paste resized image onto canvas (left-aligned)
        canvas[:, :new_w] = image
        image = canvas

        # Apply augmentations if provided
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Normalize and Invert
        # Invert: Ink becomes 1 (bright), Background becomes 0 (dark)
        # Standardize to [0, 1] float
        image = (255.0 - image) / 255.0

        # Convert to tensor (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # 3. Handle Label
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            encoded_label = self.tokenizer.encode(inchi_text)
            label_len = len(encoded_label)
            return image, encoded_label, label_len
        else:
            # For test mode, return image_id for submission creation
            image_id = row["image_id"]
            return image, image_id


def _load_dataframe(path, cache_name, load_cached_data=True):
    """
    Helper to load dataframe with caching logic.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {cache_name} from cache.")
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}")

    # 2. Compute/Load from source
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved {cache_name} to cache.")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return df


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        tokenizer (Tokenizer): Tokenizer instance.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames using caching mechanism
    df_train = _load_dataframe(
        Config.TRAIN_METADATA_PATH, "train_metadata", load_cached_data
    )
    df_val = _load_dataframe(Config.VAL_METADATA_PATH, "val_metadata", load_cached_data)
    df_test = _load_dataframe(
        Config.TEST_METADATA_PATH, "test_metadata", load_cached_data
    )

    # Debugging: Sample subset
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    # We use a reasonably large max_width to accommodate most molecules.
    # Based on EDA, max is ~2500. 2560 is a safe multiple of 32/64.
    MAX_WIDTH = 2560

    train_dataset = InChiDataset(
        df_train,
        tokenizer,
        image_height=Config.IMAGE_HEIGHT,
        max_width=MAX_WIDTH,
        mode="train",
    )

    val_dataset = InChiDataset(
        df_val,
        tokenizer,
        image_height=Config.IMAGE_HEIGHT,
        max_width=MAX_WIDTH,
        mode="val",
    )

    test_dataset = InChiDataset(
        df_test,
        tokenizer,
        image_height=Config.IMAGE_HEIGHT,
        max_width=MAX_WIDTH,
        mode="test",
    )

    # Create DataLoaders
    # Note: For variable length sequences (labels), standard collate usually works if we pad labels?
    # Actually, PyTorch's CTC loss expects a 1D tensor of concatenated labels and a separate tensor of lengths.
    # However, standard DataLoader requires stacked tensors.
    # We need a custom collate function to handle variable length targets if we want to stack them,
    # OR we can just return the 1D tensor and let PyTorch pad it?
    # Standard default_collate fails on variable length tensors.
    # We will implement a simple collate_fn here to pad the labels.

    def collate_fn(batch):
        images = []
        labels = []
        label_lengths = []
        image_ids = []

        for item in batch:
            images.append(item[0])
            if len(item) == 3:  # Train/Val
                labels.append(item[1])
                label_lengths.append(item[2])
            else:  # Test
                image_ids.append(item[1])

        # Stack images (they are fixed size due to dataset padding)
        images = torch.stack(images, 0)

        if len(labels) > 0:
            # Pad labels to max length in batch
            max_label_len = max([len(l) for l in labels])
            padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
            for i, l in enumerate(labels):
                padded_labels[i, : len(l)] = l

            label_lengths = torch.tensor(label_lengths, dtype=torch.long)
            return images, padded_labels, label_lengths
        else:
            return images, image_ids

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
