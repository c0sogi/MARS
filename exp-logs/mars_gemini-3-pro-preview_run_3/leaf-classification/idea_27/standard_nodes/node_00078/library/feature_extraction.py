import os
import numpy as np
import torch
import timm
import logging
from torch.utils.data import DataLoader, TensorDataset
from library.config import Config
from library.utils import setup_logging
from library.image_processing import process_dataset_images

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class DualStreamExtractor:
    """
    Manages the deep learning inference streams for DINOv2 and ConvNeXt.
    """

    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE

        logger.info(f"Initializing DualStreamExtractor on {self.device}...")

        # Initialize DINOv2 (Global Geometry Stream)
        logger.info(f"Loading DINOv2: {Config.MODEL_DINO}")
        self.dino_model = timm.create_model(
            Config.MODEL_DINO,
            pretrained=True,
            num_classes=0,  # Get feature embeddings
            img_size=Config.IMAGE_SIZE,
        ).to(self.device)
        self.dino_model.eval()

        # Initialize ConvNeXt (Local Texture Stream)
        logger.info(f"Loading ConvNeXt: {Config.MODEL_CONVNEXT}")
        self.convnext_model = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        ).to(self.device)
        self.convnext_model.eval()

        # Normalization constants for ImageNet
        self.mean = torch.tensor(Config.MEAN).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(Config.STD).view(1, 3, 1, 1).to(self.device)

    def preprocess_batch(self, images):
        """
        Converts a batch of numpy images to normalized tensors.

        Args:
            images (torch.Tensor): Batch of shape (N, H, W, 3) or (N, 4, H, W, 3) in uint8.

        Returns:
            torch.Tensor: Normalized batch of shape (N, 3, H, W) float32.
        """
        # If input is (B, 4, H, W, 3), flatten to (B*4, H, W, 3)
        if images.dim() == 5:
            b, v, h, w, c = images.shape
            images = images.reshape(b * v, h, w, c)

        # Permute to (N, 3, H, W)
        images = images.permute(0, 3, 1, 2).float()

        # Scale to [0, 1]
        images = images / 255.0

        # Normalize
        images = (images - self.mean) / self.std

        return images

    def process_batch(self, images_batch):
        """
        Extracts features for a batch of 4-view images and computes the canonical centroid.

        Args:
            images_batch (torch.Tensor): Batch of shape (B, 4, H, W, 3) uint8.

        Returns:
            tuple: (dino_embeddings, conv_embeddings)
                dino_embeddings: (B, D_dino)
                conv_embeddings: (B, D_conv)
        """
        batch_size = images_batch.shape[0]

        # Move to device
        images_batch = images_batch.to(self.device)

        # Preprocess (Flatten views, Normalize) -> (B*4, 3, H, W)
        processed_input = self.preprocess_batch(images_batch)

        with torch.no_grad():
            # Stream 1: DINOv2
            # Use mixed precision if available (A100 supports it well)
            with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
                dino_raw = self.dino_model(processed_input)  # (B*4, D1)
                conv_raw = self.convnext_model(processed_input)  # (B*4, D2)

        # Reshape to (B, 4, D)
        dino_views = dino_raw.view(batch_size, 4, -1)
        conv_views = conv_raw.view(batch_size, 4, -1)

        # Compute Canonical Centroid (Element-wise Average over views)
        dino_centroid = dino_views.mean(dim=1)  # (B, D1)
        conv_centroid = conv_views.mean(dim=1)  # (B, D2)

        return dino_centroid.float().cpu().numpy(), conv_centroid.float().cpu().numpy()


def extract_features(metadata_path, dataset_key, load_cached_data=True):
    """
    Orchestrates the feature extraction pipeline with caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        dataset_key (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (dino_features, conv_features, ids)
    """
    cache_dir = Config.WORKING_DIR
    dino_path = os.path.join(cache_dir, f"{dataset_key}_dino_features.npy")
    conv_path = os.path.join(cache_dir, f"{dataset_key}_conv_features.npy")
    ids_path = os.path.join(cache_dir, f"{dataset_key}_feature_ids.npy")

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(dino_path)
        and os.path.exists(conv_path)
        and os.path.exists(ids_path)
    ):
        logger.info(f"Loading cached features for {dataset_key} from {cache_dir}")
        try:
            dino_feats = np.load(dino_path)
            conv_feats = np.load(conv_path)
            ids = np.load(ids_path)
            return dino_feats, conv_feats, ids
        except Exception as e:
            logger.warning(f"Failed to load feature cache: {e}. Recomputing...")

    # 2. Load/Process Images
    # This uses image_processing.py to get the (N, 4, H, W, 3) array
    # It handles its own caching of the raw image arrays
    images, ids = process_dataset_images(
        metadata_path, dataset_key, load_cached_data=load_cached_data
    )

    if len(images) == 0:
        logger.error(f"No images found for {dataset_key}. Returning empty arrays.")
        return np.array([]), np.array([]), np.array([])

    logger.info(
        f"Starting feature extraction for {len(images)} samples ({dataset_key})..."
    )

    # 3. Initialize Extractor
    extractor = DualStreamExtractor()

    # 4. Batch Inference
    # Convert numpy array to TensorDataset for easy batching
    # Images are uint8 (N, 4, H, W, 3)
    dataset = TensorDataset(torch.from_numpy(images))
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_dino = []
    all_conv = []

    for i, batch in enumerate(loader):
        # batch is a list [tensor], unpack it
        imgs = batch[0]

        dino_emb, conv_emb = extractor.process_batch(imgs)

        all_dino.append(dino_emb)
        all_conv.append(conv_emb)

        if (i + 1) % 5 == 0:
            logger.info(f"Processed batch {i + 1}/{len(loader)}")

    # Concatenate results
    dino_feats = np.concatenate(all_dino, axis=0)
    conv_feats = np.concatenate(all_conv, axis=0)

    # 5. Save to Cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(dino_path, dino_feats)
    np.save(conv_path, conv_feats)
    np.save(ids_path, ids)

    logger.info(f"Feature extraction complete. Saved to {cache_dir}")
    logger.info(f"DINO Shape: {dino_feats.shape}, ConvNeXt Shape: {conv_feats.shape}")

    return dino_feats, conv_feats, ids
