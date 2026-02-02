import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from library.config import Config


class PetDataset(Dataset):
    """
    PyTorch Dataset for Pet Pawpularity.

    Handles loading images, applying backbone-specific preprocessing via AutoImageProcessor,
    and returning metadata and targets. Supports returning flipped images for
    feature-space augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor: AutoImageProcessor,
        input_dir: str,
        return_flip: bool = False,
        include_target: bool = True,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            processor (AutoImageProcessor): HuggingFace processor for the specific backbone.
            input_dir (str): Root directory for images (e.g., './input').
            return_flip (bool): If True, returns both original and horizontally flipped images.
            include_target (bool): If True, returns the target variable.
        """
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.input_dir = input_dir
        self.return_flip = return_flip
        self.include_target = include_target

        # Pre-fetch column indices/names for speed
        self.file_path_col = Config.FILE_PATH_COL
        self.id_col = Config.ID_COL
        self.meta_cols = Config.METADATA_COLS
        self.target_col = Config.TARGET_COL

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative (e.g., "train/xyz.jpg")
        # input_dir is "./input"
        img_path = os.path.join(self.input_dir, row[self.file_path_col])

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for missing images (should not happen given checks)
            # Create a black image of standard size (224x224 is arbitrary base, processor handles resize)
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 2. Process Image (Original)
        # return_tensors="pt" returns a dict with 'pixel_values' of shape (1, C, H, W)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)

        result = {"pixel_values": pixel_values, "id": row[self.id_col]}

        # 3. Process Image (Flipped) - Feature Space Augmentation
        if self.return_flip:
            image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            inputs_flip = self.processor(images=image_flip, return_tensors="pt")
            pixel_values_flip = inputs_flip["pixel_values"].squeeze(0)
            result["pixel_values_flip"] = pixel_values_flip

        # 4. Metadata Features
        # Convert binary columns to a float tensor
        meta_features = row[self.meta_cols].values.astype(np.float32)
        result["meta_features"] = torch.tensor(meta_features, dtype=torch.float32)

        # 5. Target
        if self.include_target and self.target_col in row:
            target = row[self.target_col]
            result["target"] = torch.tensor(target, dtype=torch.float32)

        return result


def get_loader(
    df: pd.DataFrame,
    backbone_name: str,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = Config.NUM_WORKERS,
    return_flip: bool = False,
    sample_size: int = None,
) -> DataLoader:
    """
    Factory function to create a DataLoader for a specific backbone.

    Args:
        df (pd.DataFrame): The dataframe containing data.
        backbone_name (str): Key from Config.BACKBONES (e.g., 'siglip', 'dinov2', 'convnext').
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads.
        return_flip (bool): Whether to return flipped images for augmentation.
        sample_size (int, optional): If provided, subsets the data for debugging.

    Returns:
        DataLoader: PyTorch DataLoader.
    """

    # 1. Configuration Lookup
    if backbone_name not in Config.BACKBONES:
        raise ValueError(f"Backbone '{backbone_name}' not found in Config.BACKBONES")

    backbone_config = Config.BACKBONES[backbone_name]
    model_name = backbone_config["model_name"]

    # 2. Initialize Processor
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load AutoImageProcessor for {model_name}: {e}")

    # 3. Debug Sampling
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)

    # 4. Create Dataset
    # Check if target column exists in df to set include_target
    include_target = Config.TARGET_COL in df.columns

    dataset = PetDataset(
        df=df,
        processor=processor,
        input_dir=Config.INPUT_DIR,
        return_flip=return_flip,
        include_target=include_target,
    )

    # 5. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return loader
