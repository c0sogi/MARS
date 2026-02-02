import os
import torch
import numpy as np
from transformers import SiglipVisionModel, Dinov2Model, ConvNextModel, AutoModel
from library.config import Config
from library.data_handler import get_dataloader
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles loading pre-trained backbones and extracting features from images.
    Supports Feature-Space Augmentation (averaging original and flipped views)
    and caching of extracted features to disk.
    """

    def __init__(self, backbone_name):
        """
        Args:
            backbone_name (str): HuggingFace model identifier (e.g., Config.MODEL_SIGLIP).
        """
        self.backbone_name = backbone_name
        self.device = Config.DEVICE
        self.model = self._load_model(backbone_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, backbone_name):
        """
        Loads the specific model architecture based on the name.
        """
        try:
            if "siglip" in backbone_name.lower():
                return SiglipVisionModel.from_pretrained(backbone_name)
            elif "dinov2" in backbone_name.lower():
                return Dinov2Model.from_pretrained(backbone_name)
            elif "convnext" in backbone_name.lower():
                return ConvNextModel.from_pretrained(backbone_name)
            else:
                # Fallback for other potential models
                return AutoModel.from_pretrained(backbone_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {backbone_name}: {e}")

    def extract(self, dataloader):
        """
        Runs inference on the dataloader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader yielding batches.

        Returns:
            tuple: (features, ids, meta, targets)
                - features: np.ndarray of shape (N, embedding_dim)
                - ids: np.ndarray of shape (N,)
                - meta: np.ndarray of shape (N, n_meta_features)
                - targets: np.ndarray of shape (N,) or None if targets not present
        """
        all_features = []
        all_ids = []
        all_meta = []
        all_targets = []
        has_targets = True

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                ids = batch["id"]
                pixel_values = batch["pixel_values"].to(self.device)
                meta_features = batch["meta_features"].numpy()

                # Check for targets
                if "target" in batch:
                    targets = batch["target"].numpy()
                    all_targets.append(targets)
                else:
                    has_targets = False

                # Store metadata and IDs
                all_ids.extend(ids)
                all_meta.append(meta_features)

                # Handle Feature-Space Augmentation
                # If augment=True, shape is (B, 2, C, H, W)
                # If augment=False, shape is (B, C, H, W)
                if pixel_values.ndim == 5:
                    bs, n_crops, c, h, w = pixel_values.shape
                    # Flatten to (B*2, C, H, W) for batch inference
                    inputs = pixel_values.view(-1, c, h, w)

                    # Forward pass
                    outputs = self.model(pixel_values=inputs)
                    embeddings = outputs.pooler_output  # (B*2, Dim)

                    # Reshape back to (B, 2, Dim) and average
                    embedding_dim = embeddings.shape[1]
                    embeddings = embeddings.view(bs, n_crops, embedding_dim)
                    embeddings = embeddings.mean(dim=1)  # (B, Dim)
                else:
                    # Standard inference
                    outputs = self.model(pixel_values=pixel_values)
                    embeddings = outputs.pooler_output  # (B, Dim)

                all_features.append(embeddings.cpu().numpy())

        # Concatenate all batches
        features_arr = np.concatenate(all_features, axis=0)
        ids_arr = np.array(all_ids)
        meta_arr = np.concatenate(all_meta, axis=0)

        targets_arr = None
        if has_targets and all_targets:
            targets_arr = np.concatenate(all_targets, axis=0)

        return features_arr, ids_arr, meta_arr, targets_arr

    def extract_and_cache(self, split_name, metadata_path, load_cached_data=True):
        """
        Extracts features and caches them to disk as .npy files.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            metadata_path (str): Path to the metadata CSV.
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            tuple: (features, ids, meta, targets)
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache paths
        # Config.get_cache_path returns "..._features.npy"
        # We derive other paths from it
        feature_path = Config.get_cache_path(
            self.backbone_name.split("/")[-1], split_name
        )
        base_path = feature_path.replace("_features.npy", "")

        id_path = f"{base_path}_ids.npy"
        meta_path = f"{base_path}_meta.npy"
        target_path = f"{base_path}_targets.npy"

        # Check if cache exists
        cache_exists = (
            os.path.exists(feature_path)
            and os.path.exists(id_path)
            and os.path.exists(meta_path)
        )
        # Target path might not exist for test set, so we check conditionally if needed,
        # but for simplicity, if features exist, we assume cache is valid.

        if load_cached_data and cache_exists:
            # print(f"Loading cached features for {self.backbone_name} - {split_name}")
            features = np.load(feature_path)
            ids = np.load(id_path)
            meta = np.load(meta_path)

            targets = None
            if os.path.exists(target_path):
                targets = np.load(target_path)

            return features, ids, meta, targets

        # print(f"Extracting features for {self.backbone_name} - {split_name}")

        # Create DataLoader
        loader = get_dataloader(
            metadata_path=metadata_path,
            model_name=self.backbone_name,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,  # Order must match metadata for ID alignment
            augment=Config.USE_FLIP_AUGMENTATION,
            num_workers=Config.NUM_WORKERS,
        )

        # Extract
        features, ids, meta, targets = self.extract(loader)

        # Save to cache
        np.save(feature_path, features)
        np.save(id_path, ids)
        np.save(meta_path, meta)

        if targets is not None:
            np.save(target_path, targets)
        elif os.path.exists(target_path):
            # Clean up old target file if it exists but current run has no targets (e.g. test)
            os.remove(target_path)

        return features, ids, meta, targets
