import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from library.config import Config
from library.utils import seed_everything
from library.data_loader import LeafImageDataset


class ImageEmbedder:
    """
    Extracts deep features from images using a pre-trained ResNet50 backbone.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load ResNet50 with ImageNet weights
        # Using the weights API compatible with torchvision 0.23.0
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)

        # Remove the classification head (fc layer)
        # ResNet50 forward: ... -> avgpool -> flatten -> fc
        # Replacing fc with Identity gives us the flattened 2048-d vector
        self.model.fc = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

    def extract_features(
        self, metadata_df, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    ):
        """
        Extracts features for all images in the metadata DataFrame.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'file_path' and 'id'.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.

        Returns:
            tuple: (features, ids)
                features (np.ndarray): Shape (N, 2048)
                ids (np.ndarray): Shape (N,)
        """
        dataset = LeafImageDataset(metadata_df)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        all_features = []
        all_ids = []

        with torch.no_grad():
            for images, _, ids in loader:
                images = images.to(self.device)

                # Forward pass
                # Output of resnet50 with fc=Identity is (Batch, 2048)
                outputs = self.model(images)

                # Move to CPU and numpy
                outputs = outputs.cpu().numpy()
                ids = ids.numpy()

                all_features.append(outputs)
                all_ids.append(ids)

        if not all_features:
            return np.array([]), np.array([])

        return np.concatenate(all_features, axis=0), np.concatenate(all_ids, axis=0)


def get_raw_image_features(split, load_cached_data=True):
    """
    Retrieves raw image embeddings (2048-dim) for a specific split.
    Handles caching to .npy files in the configured cache directory.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (features, ids)
    """
    # Determine paths based on split
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_filename = Config.CACHE_TRAIN_IMG_FEATS
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_filename = Config.CACHE_VAL_IMG_FEATS
    elif split == "test":
        metadata_path = Config.TEST_METADATA_PATH
        cache_filename = Config.CACHE_TEST_IMG_FEATS
    else:
        raise ValueError(f"Invalid split: {split}")

    cache_path_feats = os.path.join(Config.CACHE_DIR, cache_filename)
    cache_path_ids = os.path.join(Config.CACHE_DIR, f"{split}_ids.npy")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load metadata to verify cache consistency
    df = pd.read_csv(metadata_path)
    expected_count = len(df)

    # Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_path_feats) and os.path.exists(cache_path_ids):
            # Check if cache matches expected data length
            cached_ids = np.load(cache_path_ids)
            if len(cached_ids) == expected_count:
                print(
                    f"Loading raw image features for {split} from cache: {cache_path_feats}"
                )
                features = np.load(cache_path_feats)
                ids = cached_ids

                # Apply DEBUG limit if needed
                if Config.DEBUG:
                    return (
                        features[: Config.DEBUG_SAMPLE_SIZE],
                        ids[: Config.DEBUG_SAMPLE_SIZE],
                    )

                return features, ids
            else:
                print(
                    f"Cache mismatch for {split} (Expected {expected_count}, got {len(cached_ids)}). Recomputing..."
                )

    # Compute from scratch
    print(f"Extracting raw image features for {split}...")

    # Extract features on full dataset (caching the full set is safer)
    embedder = ImageEmbedder()
    features, ids = embedder.extract_features(df)

    # Save to cache
    print(f"Saving features to {cache_path_feats}")
    np.save(cache_path_feats, features)
    np.save(cache_path_ids, ids)

    # Apply DEBUG limit if needed
    if Config.DEBUG:
        return features[: Config.DEBUG_SAMPLE_SIZE], ids[: Config.DEBUG_SAMPLE_SIZE]

    return features, ids


def reduce_dimensions(
    train_feats,
    val_feats,
    test_feats,
    variance_threshold=Config.PCA_VARIANCE_THRESHOLD,
    load_cached_data=True,
):
    """
    Fits PCA on training features and transforms all sets.
    Handles caching of PCA components and mean.

    Args:
        train_feats (np.ndarray): Training features.
        val_feats (np.ndarray): Validation features.
        test_feats (np.ndarray): Test features.
        variance_threshold (float): Variance to retain (0.0 to 1.0).
        load_cached_data (bool): Whether to load PCA model from cache.

    Returns:
        tuple: (train_pca, val_pca, test_pca)
    """
    cache_path_components = os.path.join(Config.CACHE_DIR, Config.CACHE_PCA_COMPONENTS)
    cache_path_mean = os.path.join(Config.CACHE_DIR, Config.CACHE_PCA_MEAN)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    pca_loaded = False
    components = None
    mean = None

    # Try loading PCA model from cache
    if load_cached_data:
        if os.path.exists(cache_path_components) and os.path.exists(cache_path_mean):
            print("Loading PCA model from cache...")
            components = np.load(cache_path_components)
            mean = np.load(cache_path_mean)
            pca_loaded = True

    # Fit PCA if not loaded
    if not pca_loaded:
        print(f"Fitting PCA (variance_threshold={variance_threshold})...")
        pca = PCA(n_components=variance_threshold, random_state=Config.SEED)
        pca.fit(train_feats)

        components = pca.components_
        mean = pca.mean_

        # Save to cache
        print(f"Saving PCA components to {cache_path_components}")
        np.save(cache_path_components, components)
        np.save(cache_path_mean, mean)

    # Apply transform manually to ensure consistency with loaded arrays
    # Transform: (X - mean) @ components.T
    print("Applying PCA transform...")

    def apply_pca(X, mean_vec, comp_mat):
        X_centered = X - mean_vec
        return np.dot(X_centered, comp_mat.T)

    train_pca = apply_pca(train_feats, mean, components)
    val_pca = apply_pca(val_feats, mean, components)
    test_pca = apply_pca(test_feats, mean, components)

    return train_pca, val_pca, test_pca
