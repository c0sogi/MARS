import os
import torch
import numpy as np
from transformers import (
    CLIPVisionModel,
    AutoModel,
    ConvNextModel,
    logging as hf_logging,
)

from library.config import Config
from library.utils import get_device

# Suppress verbose HuggingFace warnings
hf_logging.set_verbosity_error()


class FeatureExtractor:
    """
    Extracts deep learning features from images using pre-trained backbones.
    Implements Feature-Space Augmentation by averaging embeddings of original
    and horizontally flipped images.
    """

    def __init__(self, backbone_key, device):
        """
        Args:
            backbone_key (str): Key matching Config.BACKBONES (e.g., 'clip', 'dinov2').
            device (torch.device): Device to run the model on.
        """
        self.device = device
        self.backbone_key = backbone_key

        if backbone_key not in Config.BACKBONES:
            raise ValueError(f"Unknown backbone key: {backbone_key}")

        self.config = Config.BACKBONES[backbone_key]
        self.model_name = self.config["name"]
        self.model_type = self.config["type"]

        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """Loads the specific HuggingFace model based on type."""
        print(f"Loading model: {self.model_name} ({self.model_type})...")
        try:
            if self.model_type == "clip":
                # CLIP: Use Vision Model only
                return CLIPVisionModel.from_pretrained(self.model_name)
            elif self.model_type == "vit":
                # DINOv2 (ViT architecture)
                return AutoModel.from_pretrained(self.model_name)
            elif self.model_type == "cnn":
                # ConvNeXt
                return ConvNextModel.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def _forward_batch(self, images):
        """
        Performs a forward pass to extract embeddings.
        Handles architecture-specific output formats.
        """
        with torch.no_grad():
            outputs = self.model(images)

            if self.model_type == "clip":
                # CLIPVisionModel returns 'pooler_output' (projected embedding)
                return outputs.pooler_output
            elif self.model_type == "vit":
                # ViT/DINOv2: Use CLS token (index 0) from last_hidden_state
                # Shape: (Batch, Seq, Hidden) -> (Batch, Hidden)
                return outputs.last_hidden_state[:, 0, :]
            elif self.model_type == "cnn":
                # ConvNextModel returns 'pooler_output' (Global Avg Pool + LayerNorm)
                return outputs.pooler_output
            else:
                raise ValueError("Unknown model type in forward pass")

    def extract(self, dataloader):
        """
        Iterates through the dataloader to extract features.
        Applies Test Time Augmentation (TTA) by flipping images.

        Returns:
            tuple: (features, ids, meta, targets) as numpy arrays.
        """
        print(f"Starting feature extraction for {self.backbone_key}...")

        features_list = []
        ids_list = []
        meta_list = []
        targets_list = []

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            # Metadata and targets are kept on CPU for storage
            meta = batch["meta"].numpy()
            targets = batch["target"].numpy()
            ids = batch["id"]  # List of strings

            # 1. Forward Pass: Original Images
            emb_orig = self._forward_batch(images)

            # 2. Forward Pass: Flipped Images (Feature-Space Augmentation)
            # Flip along width dimension (N, C, H, W) -> dim 3
            images_flip = torch.flip(images, dims=[3])
            emb_flip = self._forward_batch(images_flip)

            # 3. Average Embeddings
            emb_avg = (emb_orig + emb_flip) / 2.0

            # Store results
            features_list.append(emb_avg.cpu().numpy())
            ids_list.extend(ids)
            meta_list.append(meta)
            targets_list.append(targets)

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches...")

        # Concatenate all batches
        features = np.concatenate(features_list, axis=0)
        meta = np.concatenate(meta_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        ids = np.array(ids_list)

        print(f"Extraction complete. Features shape: {features.shape}")
        return features, ids, meta, targets


def run_extraction(dataloader, backbone_key, split, view_mode, load_cached_data=True):
    """
    Orchestrates the feature extraction process with caching.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing images.
        backbone_key (str): Key from Config.BACKBONES (e.g., 'clip').
        split (str): Dataset split name ('train', 'val', 'test').
        view_mode (str): Image view mode ('warped', 'preserved').
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (features, ids, meta, targets)
    """
    # Ensure working directory exists (handled in Config, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct unique filenames for caching
    # Format: {split}_{view}_{backbone}_{type}.npy
    base_name = f"{split}_{view_mode}_{backbone_key}"

    feat_path = os.path.join(Config.WORKING_DIR, f"{base_name}_features.npy")
    ids_path = os.path.join(Config.WORKING_DIR, f"{base_name}_ids.npy")
    meta_path = os.path.join(Config.WORKING_DIR, f"{base_name}_meta.npy")
    targets_path = os.path.join(Config.WORKING_DIR, f"{base_name}_targets.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(feat_path)
            and os.path.exists(ids_path)
            and os.path.exists(meta_path)
            and os.path.exists(targets_path)
        ):

            print(f"Loading cached features for {base_name}...")
            features = np.load(feat_path)
            ids = np.load(ids_path)
            meta = np.load(meta_path)
            targets = np.load(targets_path)
            return features, ids, meta, targets
        else:
            print(f"Cache miss for {base_name}. Starting extraction...")
    else:
        print(f"Forcing extraction for {base_name} (Cache loading disabled)...")

    # 2. Perform Extraction
    device = get_device()
    extractor = FeatureExtractor(backbone_key, device)
    features, ids, meta, targets = extractor.extract(dataloader)

    # 3. Save to cache
    print(f"Saving features to {Config.WORKING_DIR}...")
    np.save(feat_path, features)
    np.save(ids_path, ids)
    np.save(meta_path, meta)
    np.save(targets_path, targets)

    return features, ids, meta, targets
