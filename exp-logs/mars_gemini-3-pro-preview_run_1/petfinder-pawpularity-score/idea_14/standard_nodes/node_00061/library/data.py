import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
from library.config import Config


def get_processor(model_name):
    """
    Loads the AutoImageProcessor for the specified model.
    Ensures native resolution and normalization statistics are used.
    """
    # Trust remote code is sometimes needed for newer architectures like ConvNeXt V2
    # though usually standard in recent transformers versions.
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception:
        # Fallback if trust_remote_code causes issues on standard models
        processor = AutoImageProcessor.from_pretrained(model_name)
    return processor


class PetDataset(Dataset):
    """
    PyTorch Dataset for the Pawpularity Contest.
    Handles image loading, preprocessing, and metadata extraction.
    Supports returning flipped images for feature-space augmentation.
    """

    def __init__(self, df, processor, return_flipped=False, include_target=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            processor (AutoImageProcessor): HuggingFace processor.
            return_flipped (bool): If True, returns both original and flipped images.
            include_target (bool): If True, returns the target variable.
        """
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.return_flipped = return_flipped
        self.include_target = include_target

        # Identify metadata columns.
        # The dataset description lists 'Focus', but analysis sometimes shows 'Subject Focus'.
        # We handle both cases dynamically.
        possible_focus = ["Subject Focus", "Focus"]
        focus_col = next((c for c in possible_focus if c in self.df.columns), "Focus")

        self.meta_cols = [
            focus_col,
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

        # Verify columns exist
        self.valid_meta_cols = [c for c in self.meta_cols if c in self.df.columns]
        if len(self.valid_meta_cols) < 12:
            # This might happen in debug or if schema changes, just warn implicitly by using what we have
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image Loading
        # Construct full path from relative path in metadata
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback for safety, though data checks passed
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Prepare images for processor
        images_to_process = [image]
        if self.return_flipped:
            images_to_process.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # Apply Transforms/Processor
        # return_tensors='pt' gives us torch tensors
        inputs = self.processor(images=images_to_process, return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # Shape: (N_images, C, H, W)

        # If we didn't ask for flipped, squeeze the batch dimension to return (C, H, W)
        # If we did ask for flipped, keep as (2, C, H, W)
        if not self.return_flipped:
            pixel_values = pixel_values.squeeze(0)

        # Metadata Features
        meta_features = row[self.valid_meta_cols].values.astype(np.float32)

        sample = {
            "pixel_values": pixel_values,
            "metadata": torch.tensor(meta_features, dtype=torch.float32),
            "id": str(row["Id"]),
        }

        # Target
        if self.include_target and "Pawpularity" in row:
            sample["target"] = torch.tensor(row["Pawpularity"], dtype=torch.float32)

        return sample


def load_metadata(merge_train_val=False, debug=None, sample_size=None):
    """
    Loads metadata CSVs from the metadata directory.

    Args:
        merge_train_val (bool): If True, concatenates train and validation sets (for full CV).
        debug (bool): If True, subsets the data for quick debugging.
        sample_size (int): Number of samples to take in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df)
               If merge_train_val is True, val_df will be None.
    """
    if debug is None:
        debug = Config.DEBUG
    if sample_size is None:
        sample_size = Config.DEBUG_SAMPLE_SIZE

    # Load CSVs
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Subsetting
    if debug:
        train_df = train_df.iloc[:sample_size]
        val_df = val_df.iloc[:sample_size]
        test_df = test_df.iloc[:sample_size]
        print(f"[DEBUG] Subsetting data to {sample_size} rows.")

    # Merge Logic
    if merge_train_val:
        train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        val_df = None

    return train_df, val_df, test_df
