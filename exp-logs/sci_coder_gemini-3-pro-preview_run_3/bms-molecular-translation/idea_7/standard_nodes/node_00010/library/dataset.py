import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.tokenizer import Tokenizer


class ChemicalDataset(Dataset):
    """
    PyTorch Dataset for Chemical Image to InChI translation.
    Handles image loading, resizing, normalization, and label encoding.
    """

    def __init__(self, df, tokenizer, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, InChI).
            tokenizer (Tokenizer): Instance of the tokenizer for encoding labels.
            mode (str): 'train', 'val', or 'test'. Determines return values.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            (Not used in this baseline as per requirements).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # The metadata contains relative paths like "train/0/0/0/id.png"
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # 1. Load Image
        # Read as grayscale since the analysis showed 1 channel
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (though validation showed 0 missing)
            # Create a blank black image
            image = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), dtype=np.uint8)

        # 2. Preprocessing (Resize & Normalize)
        # Resize to fixed dimensions (Width, Height) -> (512, 192)
        # Note: cv2.resize expects (width, height)
        image = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # 3. Channel Expansion
        # Expand to 3 channels for ResNet compatibility (H, W) -> (H, W, 3)
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)

        # 4. To Tensor (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        if self.mode in ["train", "val"]:
            # Encode label
            inchi_text = row["InChI"]
            label_tensor = self.tokenizer.encode(inchi_text)
            label_len = len(label_tensor)

            return image, label_tensor, label_len

        else:
            # Test mode: return image and image_id
            image_id = row["image_id"]
            return image, image_id


def get_dataloader(
    metadata_path,
    tokenizer,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    shuffle=True,
    mode="train",
    debug=Config.DEBUG,
):
    """
    Factory function to create a DataLoader.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        tokenizer (Tokenizer): Tokenizer instance.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        shuffle (bool): Whether to shuffle the data.
        mode (str): 'train', 'val', or 'test'.
        debug (bool): If True, samples a small subset of data.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Debug sampling
    if debug:
        sample_size = min(len(df), Config.DEBUG_SAMPLE_SIZE)
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
        print(f"DEBUG MODE: Sampled {len(df)} rows from {metadata_path}")

    # Create Dataset
    dataset = ChemicalDataset(df, tokenizer, mode=mode)

    # Create DataLoader
    # Note: We don't need a custom collate_fn because the tokenizer
    # already pads sequences to a fixed Config.MAX_LENGTH.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(mode == "train"),  # Drop last incomplete batch during training
    )

    return dataloader
