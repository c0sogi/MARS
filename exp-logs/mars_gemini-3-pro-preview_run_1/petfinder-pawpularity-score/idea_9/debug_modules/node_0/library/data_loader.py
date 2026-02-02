import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from library.config import Config
from library.utils import seed_everything

# ==========================================
# Normalization Constants
# ==========================================
# CLIP (OpenAI) specific mean and std
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ImageNet standard mean and std (for DINOv2, ConvNeXt)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalization_stats(backbone_name):
    """
    Returns the appropriate mean and std for the given backbone.

    Args:
        backbone_name (str): Name of the backbone (e.g., 'clip', 'dinov2', 'convnext').

    Returns:
        tuple: (mean, std)
    """
    if "clip" in backbone_name.lower():
        return CLIP_MEAN, CLIP_STD
    else:
        # Default to ImageNet stats for DINOv2 and ConvNeXt
        return IMAGENET_MEAN, IMAGENET_STD


def load_metadata(mode="train", debug=False):
    """
    Loads the metadata dataframe based on the requested mode.

    Args:
        mode (str): One of 'train', 'val', 'train_all', 'test'.
                    'train_all' merges train and val sets for full CV.
        debug (bool): If True, returns a small subset for debugging.

    Returns:
        pd.DataFrame: Loaded metadata.
    """
    if mode == "train":
        df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    elif mode == "val":
        df = pd.read_csv(Config.VAL_METADATA_PATH)
    elif mode == "train_all":
        # Merge train and val for 5-fold CV on full dataset
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    elif mode == "test":
        df = pd.read_csv(Config.TEST_METADATA_PATH)
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: train, val, train_all, test."
        )

    # Debugging: Subsample data
    if debug or Config.DEBUG:
        sample_size = min(Config.DEBUG_SAMPLE_SIZE, len(df))
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
        print(f"[DEBUG] Subsampled {mode} dataset to {len(df)} rows.")

    return df


class PetDataset(Dataset):
    """
    PyTorch Dataset for Pet Pawpularity.
    Implements the Dual-View Strategy:
    1. Global View: Full image resized to target size.
    2. Local View: Central crop (focused on subject) resized to target size.
    """

    def __init__(self, df, backbone_name, transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path', 'Id', etc.
            backbone_name (str): Name of the backbone to determine normalization stats.
            transform (callable, optional): Custom transform override. If None, builds default.
        """
        self.df = df
        self.backbone_name = backbone_name

        # Pre-fetch columns to avoid overhead in __getitem__
        self.file_paths = df["file_path"].values
        self.ids = df["Id"].values

        # Extract binary metadata features (exclude non-feature columns)
        # Columns: Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur
        exclude_cols = ["Id", "Pawpularity", "file_path"]
        self.meta_cols = [c for c in df.columns if c not in exclude_cols]
        # Ensure consistent order and type
        self.meta_features = df[self.meta_cols].values.astype(np.float32)

        # Target variable
        if "Pawpularity" in df.columns:
            self.targets = df["Pawpularity"].values.astype(np.float32)
        else:
            # For test set, use zeros
            self.targets = np.zeros(len(df), dtype=np.float32)

        # Build Transforms
        mean, std = get_normalization_stats(backbone_name)

        # Common normalization
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        # Resizing
        self.resize = transforms.Resize(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct file path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load Image
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return black image in case of error (robustness)
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))

        # -----------------------------------------------------------
        # Dual-View Generation
        # -----------------------------------------------------------

        # 1. Global View: Resize full image
        global_img = self.resize(image)
        global_tensor = self.normalize(global_img)

        # 2. Local View: Central Crop -> Resize
        # Calculate crop size based on CROP_SCALE
        w, h = image.size
        crop_dim = int(min(w, h) * Config.CROP_SCALE)

        # Perform Center Crop
        left = (w - crop_dim) / 2
        top = (h - crop_dim) / 2
        right = (w + crop_dim) / 2
        bottom = (h + crop_dim) / 2

        local_img = image.crop((left, top, right, bottom))
        local_img = self.resize(local_img)
        local_tensor = self.normalize(local_img)

        # -----------------------------------------------------------
        # Metadata & Target
        # -----------------------------------------------------------
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "id": self.ids[idx],
            "global_view": global_tensor,
            "local_view": local_tensor,
            "meta": meta,
            "target": target,
        }


def get_dataloader(mode, backbone_name, batch_size=Config.BATCH_SIZE, shuffle=None):
    """
    Factory function to create a DataLoader for a specific mode and backbone.

    Args:
        mode (str): 'train', 'val', 'train_all', 'test'.
        backbone_name (str): Backbone name for normalization.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle. Defaults to True for train, False otherwise.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Load metadata
    df = load_metadata(mode=mode)

    # Initialize Dataset
    dataset = PetDataset(df, backbone_name=backbone_name)

    # Determine shuffle behavior
    if shuffle is None:
        shuffle = True if mode in ["train", "train_all"] else False

    # Determine drop_last behavior (True for training to avoid BatchNorm issues with size 1)
    drop_last = True if mode in ["train", "train_all"] else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=drop_last,
    )

    return loader
