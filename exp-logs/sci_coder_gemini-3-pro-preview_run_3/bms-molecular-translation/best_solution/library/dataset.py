import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.tokenizer import Tokenizer


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI chemical structure recognition.
    Handles image loading, preprocessing, and text tokenization.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        transform=None,
        mode: str = "train",
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (file_path, InChI, image_id).
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        self.input_root = Config.input_root
        self.image_size = Config.image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Image Loading and Preprocessing
        file_path = os.path.join(self.input_root, row["file_path"])

        # Load as grayscale (1 channel) based on EDA
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing or corrupt images (though EDA showed 0 missing)
            # Create a black image to prevent crashing
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Normalize to [0, 1] and add channel dimension
        # Shape: (H, W) -> (H, W, 1)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)

        # Convert to Tensor and Transpose to (C, H, W)
        # Shape: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 2. Label Handling
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            # Convert text to sequence of indices
            sequence = self.tokenizer.text_to_sequence(inchi_text)
            label_tensor = torch.tensor(sequence, dtype=torch.long)
            label_len = len(sequence)

            return image, label_tensor, label_len

        elif self.mode == "test":
            image_id = row["image_id"]
            return image, image_id

        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class CollateFn:
    """
    Custom collate function to handle variable length sequences in a batch.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Check if the batch contains labels (train/val) or image_ids (test)
        # batch[0] is (image, label_tensor, label_len) OR (image, image_id)

        if len(batch[0]) == 3:  # Train/Val mode
            images, labels, lengths = zip(*batch)

            # Stack images: (B, C, H, W)
            images = torch.stack(images, 0)

            # Pad labels: (B, max_len)
            # batch_first=True makes output (B, T)
            labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_idx)

            lengths = torch.tensor(lengths, dtype=torch.long)

            return images, labels, lengths

        elif len(batch[0]) == 2:  # Test mode
            images, image_ids = zip(*batch)
            images = torch.stack(images, 0)
            return images, list(image_ids)

        else:
            raise RuntimeError("Unknown batch structure.")


def get_train_val_loaders(config: Config, tokenizer: Tokenizer):
    """
    Creates DataLoaders for training and validation sets.

    Args:
        config (Config): Configuration object.
        tokenizer (Tokenizer): Tokenizer object.

    Returns:
        train_loader, val_loader
    """
    # Load Metadata
    train_df = pd.read_csv(config.train_metadata_path)
    val_df = pd.read_csv(config.val_metadata_path)

    # Debug Mode: Sample subset
    if config.debug:
        print(
            f"[DEBUG] Sampling {config.debug_sample_size} rows for training/validation."
        )
        train_df = train_df.sample(
            n=min(len(train_df), config.debug_sample_size), random_state=config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), config.debug_sample_size), random_state=config.seed
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = InChiDataset(train_df, tokenizer, mode="train")
    val_dataset = InChiDataset(val_df, tokenizer, mode="val")

    # Collate Function with padding index
    pad_idx = tokenizer.stoi[config.pad_token]
    collate_fn = CollateFn(pad_idx=pad_idx)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(config: Config, tokenizer: Tokenizer):
    """
    Creates DataLoader for the test set.

    Args:
        config (Config): Configuration object.
        tokenizer (Tokenizer): Tokenizer object.

    Returns:
        test_loader
    """
    test_df = pd.read_csv(config.test_metadata_path)

    if config.debug:
        print(f"[DEBUG] Sampling {config.debug_sample_size} rows for testing.")
        test_df = test_df.sample(
            n=min(len(test_df), config.debug_sample_size), random_state=config.seed
        ).reset_index(drop=True)

    test_dataset = InChiDataset(test_df, tokenizer, mode="test")

    # For test loader, we don't need padding logic for labels, but we reuse the class structure
    pad_idx = tokenizer.stoi[config.pad_token]
    collate_fn = CollateFn(pad_idx=pad_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
