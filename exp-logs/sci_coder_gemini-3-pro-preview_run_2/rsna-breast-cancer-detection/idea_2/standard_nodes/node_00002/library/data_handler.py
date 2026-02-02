import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    METADATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
)
from library.utils import read_dicom_manual
from library.vision_backbone import get_image_encoder


class MammogramDataset(Dataset):
    """
    PyTorch Dataset for loading Mammogram images and preparing them for the ResNet backbone.
    """

    def __init__(self, df):
        self.df = df
        self.file_paths = df["file_path"].values

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load image using the robust manual reader
        img = read_dicom_manual(full_path)

        # Handle loading failures or missing files
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Resize to target dimension
        try:
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Convert Grayscale to RGB (3 channels) for ResNet compatibility
        # read_dicom_manual returns grayscale (H, W)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert to Tensor: (H, W, C) -> (C, H, W) and scale [0, 255] -> [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Apply normalization
        img_tensor = self.normalize(img_tensor)

        return img_tensor


def generate_embeddings(df, model, device, batch_size=BATCH_SIZE):
    """
    Generates image embeddings using the frozen backbone model.

    Args:
        df (pd.DataFrame): Metadata dataframe containing file paths.
        model (nn.Module): Frozen feature extractor.
        device (torch.device): Device to run inference on.
        batch_size (int): Batch size for DataLoader.

    Returns:
        pd.DataFrame: Original dataframe with added embedding columns.
    """
    dataset = MammogramDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    embeddings = []

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            # Forward pass: returns (Batch, 512)
            features = model(imgs)
            embeddings.append(features.cpu().numpy())

    # Concatenate all batch results
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.empty((0, 512))

    # Create a DataFrame for the embeddings
    # Naming columns emb_0 to emb_511
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)

    # Reset index to ensure clean concatenation
    df_reset = df.reset_index(drop=True)

    # Concatenate original metadata with embeddings
    result_df = pd.concat([df_reset, emb_df], axis=1)

    return result_df


def get_processed_data(load_cached_data=True):
    """
    Main data handling function. Loads metadata, generates (or loads) embeddings,
    and returns processed DataFrames for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_cache = os.path.join(WORKING_DIR, "train_embeddings.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_embeddings.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_embeddings.parquet")

    # Check for cached data
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        else:
            print("Cache missing or incomplete. Generating embeddings from scratch...")
    else:
        print("Ignoring cache. Generating embeddings from scratch...")

    # Load raw metadata
    print("Loading metadata...")
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Initialize Model
    print("Initializing vision backbone...")
    model = get_image_encoder()

    # Generate Embeddings
    print(f"Processing Training Set ({len(train_meta)} images)...")
    train_df = generate_embeddings(train_meta, model, DEVICE)

    print(f"Processing Validation Set ({len(val_meta)} images)...")
    val_df = generate_embeddings(val_meta, model, DEVICE)

    print(f"Processing Test Set ({len(test_meta)} images)...")
    test_df = generate_embeddings(test_meta, model, DEVICE)

    # Save to Cache
    print("Saving processed data to parquet cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
