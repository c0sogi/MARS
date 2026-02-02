import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from library import config


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class FeatureExtractor:
    def __init__(self):
        self.device = config.DEVICE
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        """
        Loads ConvNeXt-Large and modifies the head to return embeddings.
        """
        weights = getattr(models.ConvNeXt_Large_Weights, config.WEIGHTS)
        model = models.convnext_large(weights=weights)

        # The classifier is a Sequential block:
        # [0] LayerNorm2d, [1] Flatten, [2] Linear
        # We replace the Linear layer with Identity to keep GAP and LayerNorm.
        model.classifier[2] = nn.Identity()

        return model

    def _extract_view_features(self, images):
        """
        Performs forward pass with Test Time Augmentation (Horizontal Flip).
        """
        # Forward pass on original images
        features_orig = self.model(images)

        if config.USE_TTA_FLIP:
            # Flip images horizontally (dim 3 is width)
            images_flipped = torch.flip(images, dims=[3])
            features_flip = self.model(images_flipped)
            # Average the embeddings
            return (features_orig + features_flip) / 2.0

        return features_orig

    def extract(self, data_loader):
        """
        Iterates over the dataloader, extracts features for all 3 views,
        and performs Early Fusion (concatenation).
        """
        all_embeddings = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for batch in data_loader:
                # Move inputs to device
                view_global = batch["global"].to(self.device)
                view_standard = batch["standard"].to(self.device)
                view_local = batch["local"].to(self.device)

                ids = batch["id"]

                # Extract features for each view (with TTA)
                emb_global = self._extract_view_features(view_global)
                emb_standard = self._extract_view_features(view_standard)
                emb_local = self._extract_view_features(view_local)

                # Early Fusion: Concatenate features
                # Result shape: (Batch, 1536 * 3) = (Batch, 4608)
                fused_emb = torch.cat([emb_global, emb_standard, emb_local], dim=1)

                # Store results
                all_embeddings.append(fused_emb.cpu().numpy())
                all_ids.extend(ids)

                if "label" in batch:
                    all_labels.append(batch["label"].numpy())

        # Aggregate into final arrays
        embeddings = np.vstack(all_embeddings)
        ids = np.array(all_ids)

        if all_labels:
            labels = np.concatenate(all_labels)
        else:
            labels = None

        return embeddings, labels, ids


def extract_features(data_loader, dataset_name, load_cached_data=True):
    """
    Extracts features for the given dataset, utilizing caching to avoid re-computation.

    Args:
        data_loader: The DataLoader for the dataset.
        dataset_name: 'train', 'val', or 'test' (used for cache filenames).
        load_cached_data: If True, attempts to load from disk first.

    Returns:
        embeddings (np.ndarray), labels (np.ndarray or None), ids (np.ndarray)
    """
    set_seed(config.SEED)

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    emb_path = os.path.join(config.WORKING_DIR, f"{dataset_name}_embeddings.npy")
    lbl_path = os.path.join(config.WORKING_DIR, f"{dataset_name}_labels.npy")
    id_path = os.path.join(config.WORKING_DIR, f"{dataset_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        # Check if essential files exist
        if os.path.exists(emb_path) and os.path.exists(id_path):
            # For train/val, labels must also exist. For test, they might not.
            labels_exist = os.path.exists(lbl_path)
            if dataset_name == "test" or labels_exist:
                print(f"Loading cached features for {dataset_name}...")
                embeddings = np.load(emb_path)
                ids = np.load(id_path)
                labels = np.load(lbl_path) if labels_exist else None
                return embeddings, labels, ids

    # Compute features from scratch
    print(f"Extracting features for {dataset_name}...")
    extractor = FeatureExtractor()
    embeddings, labels, ids = extractor.extract(data_loader)

    # Save to cache
    print(f"Saving features for {dataset_name} to {config.WORKING_DIR}...")
    np.save(emb_path, embeddings)
    np.save(id_path, ids)
    if labels is not None:
        np.save(lbl_path, labels)

    return embeddings, labels, ids
