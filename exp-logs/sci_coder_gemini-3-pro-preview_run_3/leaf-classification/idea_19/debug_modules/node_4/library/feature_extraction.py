import os
import numpy as np
import pandas as pd
import torch
import timm
from typing import Tuple

from library.config import Config
from library.utils import load_numpy, save_numpy, ensure_directory
from library.image_processing import load_image, generate_rotated_views


class FeatureExtractor:
    """
    Wrapper class for DINOv2 and ConvNeXt feature extraction.
    Initializes models in evaluation mode and handles batched inference on GPU.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Initializing FeatureExtractor on {self.device}...")

        # Initialize DINOv2 (Global Geometry)
        # num_classes=0 returns the pooled features/embedding
        print(f"Loading DINOv2: {Config.MODEL_DINOV2}")
        self.dino = timm.create_model(
            Config.MODEL_DINOV2,
            pretrained=True,
            num_classes=0,
            img_size=Config.IMAGE_SIZE,
        ).to(self.device)
        self.dino.eval()

        # Initialize ConvNeXt (Local Texture)
        print(f"Loading ConvNeXt: {Config.MODEL_CONVNEXT}")
        self.convnext = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        ).to(self.device)
        self.convnext.eval()

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Extracts concatenated features from a batch of images.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            np.ndarray: Features of shape (B, D_dino + D_convnext).
        """
        images = images.to(self.device)

        # Extract features from both streams
        dino_emb = self.dino(images)
        convnext_emb = self.convnext(images)

        # Concatenate along feature dimension
        # DINOv2 Large (~1024) + ConvNeXt Large (~1536) -> ~2560 dimensions
        combined = torch.cat([dino_emb, convnext_emb], dim=1)

        return combined.cpu().numpy()


def extract_dataset_features(
    metadata_path: str,
    cache_features_path: str,
    cache_ids_path: str,
    load_cached_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 36-view features for all images in the specified metadata file.
    Implements caching to avoid redundant computation.

    Args:
        metadata_path: Path to the CSV file containing image metadata (must have 'file_path' and 'id').
        cache_features_path: Path to save/load the extracted features (.npy).
        cache_ids_path: Path to save/load the corresponding IDs (.npy).
        load_cached_data: If True, attempts to load from cache first.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, ids)
            features: Shape (N, 36, Feature_Dim)
            ids: Shape (N,)
    """
    # 1. Check Cache
    if load_cached_data:
        if os.path.exists(cache_features_path) and os.path.exists(cache_ids_path):
            print(f"Loading cached features from {cache_features_path}...")
            return load_numpy(cache_features_path), load_numpy(cache_ids_path)
        else:
            print("Cache not found or incomplete. Starting extraction...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Ensure required columns exist
    if "file_path" not in df.columns or "id" not in df.columns:
        raise ValueError("Metadata CSV must contain 'file_path' and 'id' columns.")

    ids = df["id"].values
    file_paths = df["file_path"].values
    num_samples = len(df)

    # 3. Initialize Extractor
    extractor = FeatureExtractor()

    all_features = []

    print(f"Starting feature extraction for {num_samples} images...")

    # 4. Processing Loop
    for idx, rel_path in enumerate(file_paths):
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        try:
            # Load image and generate views
            # Returns (3, H, W)
            img_tensor = load_image(full_path)
            # Returns (36, 3, H, W)
            views_tensor = generate_rotated_views(img_tensor)

            # Process views in batches to respect Config.BATCH_SIZE
            # Even though 36 is small, we batch it for safety and consistency
            num_views = views_tensor.shape[0]
            img_view_features = []

            for b_start in range(0, num_views, Config.BATCH_SIZE):
                b_end = min(b_start + Config.BATCH_SIZE, num_views)
                batch_views = views_tensor[b_start:b_end]

                batch_feats = extractor.extract_batch(batch_views)
                img_view_features.append(batch_feats)

            # Concatenate all views for this image
            # Shape: (36, Feature_Dim)
            img_features = np.concatenate(img_view_features, axis=0)
            all_features.append(img_features)

        except Exception as e:
            print(f"Error processing image {full_path}: {e}")
            raise e

        # Logging
        if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
            print(f"Processed {idx + 1}/{num_samples} images.")

    # 5. Stack and Save
    # Shape: (N, 36, Feature_Dim)
    all_features_np = np.stack(all_features, axis=0)

    print(f"Saving features to {cache_features_path}...")
    save_numpy(all_features_np, cache_features_path)
    save_numpy(ids, cache_ids_path)

    print("Feature extraction complete.")
    return all_features_np, ids
