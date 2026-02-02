import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
from tqdm import tqdm
import timm
from transformers import AutoModel
from torchvision import transforms
from torchvision.transforms import functional as F

from library import config, utils

# Set up logging
logger = utils.setup_logger(
    "feature_extractor", os.path.join(config.WORKING_DIR, "feature_extraction.log")
)


class LeafFeatureExtractor:
    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Initializing LeafFeatureExtractor on {self.device}...")

        # 1. Initialize DINOv2 (Global Geometry Stream)
        logger.info(f"Loading DINOv2 model: {config.MODEL_DINO}")
        self.dino_model = AutoModel.from_pretrained(config.MODEL_DINO).to(self.device)
        self.dino_model.eval()

        # 2. Initialize ConvNeXt (Local Texture Stream)
        logger.info(f"Loading ConvNeXt model: {config.MODEL_CONVNEXT}")
        self.conv_model = timm.create_model(
            config.MODEL_CONVNEXT,
            pretrained=True,
            num_classes=0,  # Use as feature extractor (pooling is included by default in timm for num_classes=0)
        ).to(self.device)
        self.conv_model.eval()

        # Preprocessing definitions
        # ImageNet mean and std
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Base resize transform
        self.resize = transforms.Resize(
            config.IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def preprocess_image(self, image_path):
        """
        Loads an image, resizes it, and returns a PIL Image.
        """
        # Load image using PIL (handles formats well)
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank white image in case of failure to avoid crashing
            img = Image.new("RGB", config.IMG_SIZE, (255, 255, 255))

        img = self.resize(img)
        return img

    def get_12_views(self, pil_img):
        """
        Generates 12 rotated views of the image.
        Returns a batch tensor of shape [12, 3, H, W].
        """
        views = []
        # Convert to tensor first (C, H, W) in range [0, 1]
        img_tensor = self.to_tensor(pil_img)

        for angle in config.ROTATION_ANGLES:
            # Rotate
            # fill=[1.0, 1.0, 1.0] corresponds to white background for normalized float tensor (if it was 0-255 it would be 255)
            # However, F.rotate expects fill to be a number or list.
            # Since we are operating on a Tensor [0,1], fill should be 1.0.
            rotated = F.rotate(img_tensor, angle=float(angle), fill=[1.0])

            # Normalize
            normalized = self.normalize(rotated)
            views.append(normalized)

        return torch.stack(views)  # [12, 3, H, W]

    def extract_batch(self, image_paths):
        """
        Extracts features for a batch of image paths.
        Returns:
            dino_feats: numpy array [B, 12, D_dino]
            conv_feats: numpy array [B, 12, D_conv]
        """
        batch_views = []
        valid_paths = []

        # 1. Prepare Data
        for p in image_paths:
            img = self.preprocess_image(p)
            views = self.get_12_views(img)  # [12, 3, H, W]
            batch_views.append(views)
            valid_paths.append(p)

        if not batch_views:
            return None, None

        # Stack into a single large batch: [B * 12, 3, H, W]
        # This allows parallel inference on all views
        input_tensor = torch.cat(batch_views, dim=0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            # DINOv2 Inference
            # HuggingFace ViT output.last_hidden_state is [Batch, Seq, Dim]
            # We take the CLS token (index 0)
            dino_out = self.dino_model(input_tensor).last_hidden_state[:, 0, :]

            # ConvNeXt Inference
            # timm with num_classes=0 returns the pooled feature vector [Batch, Dim]
            conv_out = self.conv_model(input_tensor)

        # 3. Reshape and Move to CPU
        # Reshape from [B*12, Dim] to [B, 12, Dim]
        batch_size = len(image_paths)

        dino_feats = dino_out.view(batch_size, config.NUM_VIEWS, -1).cpu().numpy()
        conv_feats = conv_out.view(batch_size, config.NUM_VIEWS, -1).cpu().numpy()

        return dino_feats, conv_feats


def extract_multi_view_features(metadata_df, dataset_name, load_cached_data=True):
    """
    Main function to handle feature extraction with caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {
            'ids': np.ndarray,
            'dino_features': np.ndarray,
            'conv_features': np.ndarray
        }
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    path_ids = os.path.join(cache_dir, f"{dataset_name}_ids.npy")
    path_dino = os.path.join(cache_dir, f"{dataset_name}_dino.npy")
    path_conv = os.path.join(cache_dir, f"{dataset_name}_conv.npy")

    # 1. Check Cache
    if load_cached_data:
        if (
            os.path.exists(path_ids)
            and os.path.exists(path_dino)
            and os.path.exists(path_conv)
        ):
            logger.info(
                f"Loading cached features for {dataset_name} from {cache_dir}..."
            )
            try:
                ids = utils.load_numpy(path_ids)
                dino_feats = utils.load_numpy(path_dino)
                conv_feats = utils.load_numpy(path_conv)

                # Verify consistency
                if len(ids) == len(metadata_df):
                    logger.info("Cache loaded successfully.")
                    return {
                        "ids": ids,
                        "dino_features": dino_feats,
                        "conv_features": conv_feats,
                    }
                else:
                    logger.warning(
                        f"Cache size mismatch ({len(ids)} vs {len(metadata_df)}). Recomputing..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            logger.info(f"Cache not found for {dataset_name}. Starting extraction...")
    else:
        logger.info(f"Ignoring cache for {dataset_name}. Starting extraction...")

    # 2. Initialize Extractor
    extractor = LeafFeatureExtractor()

    # 3. Processing Loop
    all_ids = []
    all_dino = []
    all_conv = []

    # Prepare paths
    # Metadata 'file_path' is relative to input dir, e.g., "images/1.jpg"
    full_paths = [
        os.path.join(config.INPUT_DIR, str(p)) for p in metadata_df["file_path"].values
    ]
    ids = metadata_df["id"].values

    batch_size = config.BATCH_SIZE
    total_images = len(full_paths)

    # Process in batches
    for i in tqdm(
        range(0, total_images, batch_size), desc=f"Extracting {dataset_name}"
    ):
        batch_paths = full_paths[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        dino_batch, conv_batch = extractor.extract_batch(batch_paths)

        if dino_batch is not None:
            all_ids.append(batch_ids)
            all_dino.append(dino_batch)
            all_conv.append(conv_batch)

    # 4. Aggregate Results
    final_ids = np.concatenate(all_ids, axis=0)
    final_dino = np.concatenate(all_dino, axis=0)
    final_conv = np.concatenate(all_conv, axis=0)

    logger.info(
        f"Extraction complete. Shapes: IDs={final_ids.shape}, DINO={final_dino.shape}, CONV={final_conv.shape}"
    )

    # 5. Save to Cache
    logger.info(f"Saving features to {cache_dir}...")
    utils.save_numpy(final_ids, path_ids)
    utils.save_numpy(final_dino, path_dino)
    utils.save_numpy(final_conv, path_conv)

    return {"ids": final_ids, "dino_features": final_dino, "conv_features": final_conv}
