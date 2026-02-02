import os
import numpy as np
import torch
import torch.nn as nn
import timm
from library.config import (
    DEVICE,
    SEED,
    TRAIN_EMBEDDINGS_PATH,
    TRAIN_LABELS_PATH,
    VAL_EMBEDDINGS_PATH,
    VAL_LABELS_PATH,
    TEST_EMBEDDINGS_PATH,
    TEST_IDS_PATH,
    MODEL_NAME,
)
from library.utils import set_seed


class FeatureExtractor:
    """
    A class to extract features from images using a pre-trained ResNet50 model.
    Handles caching of extracted features to disk to speed up subsequent runs.
    """

    def __init__(self):
        """
        Initializes the FeatureExtractor.
        Sets the random seed, loads the pre-trained model via timm,
        and moves the model to the configured device.
        """
        set_seed(SEED)

        print(f"Loading model: {MODEL_NAME}...")
        # Load model using timm
        # num_classes=0 removes the final classification head and returns the pooled features
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)

        self.model.to(DEVICE)
        self.model.eval()

    def extract(self, dataloader):
        """
        Runs inference on the dataloader to extract features.
        Implements Feature-Level Test Time Augmentation (TTA) by averaging
        embeddings of the original and horizontally flipped images.

        Args:
            dataloader (DataLoader): The torch DataLoader containing images.

        Returns:
            tuple: (features, targets)
                - features (np.ndarray): Extracted embeddings of shape (N, 1536).
                - targets (np.ndarray): Corresponding labels (int) or IDs (str).
        """
        features_list = []
        targets_list = []

        # Disable gradient calculation for inference
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(DEVICE, non_blocking=True)

                # Forward pass (Original)
                features_orig = self.model(images)

                # Forward pass (Horizontal Flip TTA)
                # dim 3 is width in (N, C, H, W)
                features_flip = self.model(torch.flip(images, dims=[3]))

                # Average the embeddings
                outputs = (features_orig + features_flip) / 2.0

                # Move to CPU and convert to numpy
                features_list.append(outputs.cpu().numpy())

                # Handle targets
                # Train/Val datasets return Tensor labels
                # Test dataset returns tuple/list of ID strings
                if isinstance(targets, torch.Tensor):
                    targets_list.append(targets.cpu().numpy())
                else:
                    targets_list.append(np.array(targets))

        if not features_list:
            return np.array([]), np.array([])

        # Concatenate all batches
        features = np.vstack(features_list)
        targets = np.concatenate(targets_list)

        return features, targets

    def _get_features_generic(
        self, dataloader, embeddings_path, targets_path, load_cached_data
    ):
        """
        Helper method to handle the check-load-compute-save logic for features.

        Args:
            dataloader (DataLoader): The data source.
            embeddings_path (str): Path to save/load embeddings .npy file.
            targets_path (str): Path to save/load targets .npy file.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (features, targets)
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(embeddings_path)
            and os.path.exists(targets_path)
        ):
            print(f"Loading cached features from {embeddings_path}")
            try:
                embeddings = np.load(embeddings_path)
                # allow_pickle=True is required for string arrays (IDs)
                targets = np.load(targets_path, allow_pickle=True)
                return embeddings, targets
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Computing features for {os.path.basename(embeddings_path)}...")
        embeddings, targets = self.extract(dataloader)

        # 3. Save to cache
        np.save(embeddings_path, embeddings)
        np.save(targets_path, targets)
        print(f"Saved features to {embeddings_path}")

        return embeddings, targets

    def get_train_features(self, dataloader, load_cached_data=True):
        """
        Returns training features and labels.
        """
        return self._get_features_generic(
            dataloader, TRAIN_EMBEDDINGS_PATH, TRAIN_LABELS_PATH, load_cached_data
        )

    def get_val_features(self, dataloader, load_cached_data=True):
        """
        Returns validation features and labels.
        """
        return self._get_features_generic(
            dataloader, VAL_EMBEDDINGS_PATH, VAL_LABELS_PATH, load_cached_data
        )

    def get_test_features(self, dataloader, load_cached_data=True):
        """
        Returns test features and IDs.
        """
        return self._get_features_generic(
            dataloader, TEST_EMBEDDINGS_PATH, TEST_IDS_PATH, load_cached_data
        )
