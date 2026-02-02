import os
import torch
import numpy as np
import timm
from library.config import Config
from library.utils import seed_everything


class FeatureExtractor:
    """
    Feature extractor using a frozen backbone from timm.
    Extracts high-dimensional embeddings from images and retrieves metadata/targets.
    """

    def __init__(self):
        """
        Initializes the model, freezes weights, and moves to the configured device.
        """
        # Ensure reproducibility
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        # Load pre-trained model using timm
        # num_classes=0 removes the classification head and returns pooled features
        print(f"Loading model: {Config.MODEL_NAME}")
        self.model = timm.create_model(
            Config.MODEL_NAME, pretrained=True, num_classes=0
        )

        # Freeze all parameters to use as a fixed feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

        # Move model to the appropriate device (CPU/GPU)
        self.model.to(self.device)
        self.model.eval()

    def extract_features(
        self,
        dataloader,
        cache_features_path,
        cache_aux_path,
        load_cached_data=True,
        is_test=False,
        max_batches=None,
    ):
        """
        Extracts features from the dataloader, with caching support.

        Args:
            dataloader: PyTorch DataLoader yielding (image, metadata, target/id).
            cache_features_path (str): Path to save/load image features (.npy).
            cache_aux_path (str): Path to save/load targets (train/val) or IDs (test) (.npy).
            load_cached_data (bool): Whether to attempt loading from cache.
            is_test (bool): Whether processing the test set (affects aux data handling).
            max_batches (int, optional): Limit the number of batches for debugging.

        Returns:
            tuple: (features, metadata, aux_data)
                - features: numpy array of shape (N, 576)
                - metadata: numpy array of shape (N, 12)
                - aux_data: numpy array of targets (N,) or IDs (N,)
        """
        # Define path for metadata cache (derived from features path)
        # We assume cache_features_path follows the pattern *_features.npy
        cache_meta_path = cache_features_path.replace("features.npy", "meta.npy")
        if cache_meta_path == cache_features_path:
            cache_meta_path = cache_features_path + ".meta.npy"

        # Ensure the working directory exists
        os.makedirs(os.path.dirname(cache_features_path), exist_ok=True)

        # 1. Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(cache_features_path)
                and os.path.exists(cache_aux_path)
                and os.path.exists(cache_meta_path)
            ):

                print(
                    f"Loading cached features from {os.path.dirname(cache_features_path)}..."
                )
                try:
                    features = np.load(cache_features_path)
                    metadata = np.load(cache_meta_path)
                    # allow_pickle=True is required for string IDs in the test set
                    aux_data = np.load(cache_aux_path, allow_pickle=True)
                    return features, metadata, aux_data
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")
            else:
                print("Cache not found. Computing features...")
        else:
            print("Forced re-computation. Computing features...")

        # 2. Compute features if not cached
        features_list = []
        meta_list = []
        aux_list = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break

                images, meta, aux = batch
                images = images.to(self.device)

                # Forward pass through the backbone
                # timm models with num_classes=0 return the pooled feature vector directly
                emb = self.model(images)

                # Collect batch results
                features_list.append(emb.cpu().numpy())
                meta_list.append(meta.numpy())

                if is_test:
                    # For test set, aux is a tuple of IDs (strings)
                    aux_list.extend(aux)
                else:
                    # For train/val, aux is a tensor of targets
                    aux_list.append(aux.numpy())

        # Handle empty dataloader case
        if not features_list:
            return np.array([]), np.array([]), np.array([])

        # Concatenate all batches
        features = np.vstack(features_list)
        metadata = np.vstack(meta_list)

        if is_test:
            aux_data = np.array(aux_list)
        else:
            aux_data = np.concatenate(aux_list)

        # 3. Save to cache
        print(f"Saving features to {cache_features_path}...")
        np.save(cache_features_path, features)
        np.save(cache_meta_path, metadata)
        np.save(cache_aux_path, aux_data)

        return features, metadata, aux_data
