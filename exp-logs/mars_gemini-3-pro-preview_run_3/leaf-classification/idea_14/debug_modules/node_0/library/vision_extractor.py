import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple

from library.configuration import Config
from library.utilities import setup_logger

# Initialize logger
logger = setup_logger()


class DualStreamExtractor(nn.Module):
    """
    A PyTorch module that encapsulates the Dual-Stream Early Fusion architecture.
    Stream 1: DINOv2 (ViT-Large) for global geometric priors.
    Stream 2: ConvNeXt Large for local texture details.
    """

    def __init__(self):
        super(DualStreamExtractor, self).__init__()

        logger.info(f"Initializing DINOv2 model: {Config.MODEL_DINO_NAME}")
        self.dino = timm.create_model(
            Config.MODEL_DINO_NAME,
            pretrained=True,
            num_classes=0,  # Remove classification head to get embeddings
        )

        logger.info(f"Initializing ConvNeXt model: {Config.MODEL_CONV_NAME}")
        self.conv = timm.create_model(
            Config.MODEL_CONV_NAME,
            pretrained=True,
            num_classes=0,  # Remove classification head to get embeddings
        )

        # Set to evaluation mode immediately
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract and concatenate features.
        Args:
            x (torch.Tensor): Input batch of images (B, 3, H, W).
        Returns:
            torch.Tensor: Concatenated embeddings (B, D_dino + D_conv).
        """
        with torch.no_grad():
            # Extract features from both streams
            # timm's num_classes=0 typically returns the pooled feature or CLS token
            dino_features = self.dino(x)
            conv_features = self.conv(x)

            # Concatenate features along the channel/feature dimension
            combined_features = torch.cat([dino_features, conv_features], dim=1)

        return combined_features


class LeafRotationDataset(Dataset):
    """
    Dataset class that loads images and generates 12 orthogonal views per image.
    """

    def __init__(self, df: pd.DataFrame, input_dir: str):
        self.df = df
        self.input_dir = input_dir
        self.rotation_angles = Config.ROTATION_ANGLES

        # Preprocessing pipeline (after rotation)
        # Note: We do ToTensor and Normalize here. Resizing happens after rotation in __getitem__
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative path 'images/{id}.jpg'
        # Input dir is './input'
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        views = []

        for angle in self.rotation_angles:
            # 1. Rotate
            # Use white border (255, 255, 255) to match background
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

            # 2. Resize to model input size
            resized_img = cv2.resize(
                rotated_img,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )

            # 3. Normalize and convert to Tensor
            # ToTensor converts [0, 255] -> [0.0, 1.0]
            tensor_img = self.normalize(resized_img)
            views.append(tensor_img)

        # Stack views: (12, 3, H, W)
        return torch.stack(views)


def extract_rotational_features(
    df: pd.DataFrame,
    subset_name: str,
    load_cached_data: bool = True,
    batch_size: int = Config.BATCH_SIZE,
    limit: Optional[int] = None,
) -> np.ndarray:
    """
    Extracts features for 12 rotated views of each image in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing 'file_path' column.
        subset_name (str): Name of the subset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from disk.
        batch_size (int): Batch size for inference.
        limit (int, optional): Limit number of samples for debugging.

    Returns:
        np.ndarray: Feature tensor of shape (N_samples, 12, Feature_Dim).
    """

    # 1. Setup Cache Path
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{subset_name}_vision_features.npy")

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached vision features from {cache_path}")
        try:
            features = np.load(cache_path)
            # If limit is applied, ensure we return the correct slice even if full cache loaded
            if limit is not None:
                return features[:limit]
            # Verify length matches dataframe
            if len(features) == len(df):
                return features
            else:
                logger.warning(
                    f"Cached features length ({len(features)}) does not match dataframe ({len(df)}). Recomputing."
                )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing.")

    # 3. Prepare Data
    if limit is not None:
        logger.info(f"Limiting dataset to {limit} samples for debugging.")
        df_proc = df.iloc[:limit].copy()
    else:
        df_proc = df

    dataset = LeafRotationDataset(df_proc, Config.INPUT_DIR)

    # DataLoader
    # Note: Dataset returns (12, 3, H, W). DataLoader yields (B, 12, 3, H, W).
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # 4. Initialize Model
    device = torch.device(Config.DEVICE)
    model = DualStreamExtractor()
    model.to(device)

    # 5. Inference Loop
    logger.info(
        f"Starting feature extraction for {len(dataset)} images ({len(dataset) * 12} total views)..."
    )

    all_features = []

    # Use mixed precision for speed and memory efficiency on A100
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for i, batch_imgs in enumerate(dataloader):
            # batch_imgs shape: (B, 12, 3, H, W)
            B, V, C, H, W = batch_imgs.shape

            # Flatten batch and views for inference: (B*12, 3, H, W)
            flat_imgs = batch_imgs.view(B * V, C, H, W).to(device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features_flat = model(flat_imgs)
            else:
                features_flat = model(flat_imgs)

            # Reshape back to (B, 12, Feature_Dim)
            # features_flat shape: (B*12, D) -> (B, 12, D)
            features_batch = features_flat.view(B, V, -1)

            all_features.append(features_batch.cpu().numpy())

            if (i + 1) % 5 == 0:
                logger.info(f"Processed batch {i + 1}/{len(dataloader)}")

    # 6. Aggregate and Save
    final_features = np.concatenate(all_features, axis=0)

    # Ensure float32 for compatibility
    final_features = final_features.astype(np.float32)

    logger.info(f"Feature extraction complete. Shape: {final_features.shape}")

    # Save to cache if we processed the full requested dataset (not just a limit slice)
    if limit is None:
        logger.info(f"Saving features to {cache_path}")
        np.save(cache_path, final_features)

    return final_features
