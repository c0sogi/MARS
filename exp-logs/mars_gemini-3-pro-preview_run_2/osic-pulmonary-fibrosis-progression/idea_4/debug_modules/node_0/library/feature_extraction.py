import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from sklearn.decomposition import PCA
from library.config import Config
from library.image_processing import process_patient_scan
from library.utils import seed_everything


class CNNFeatureExtractor:
    """
    Extracts visual features from CT scans using a pre-trained EfficientNet-B0.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model with num_classes=0 to get global pooled features (embeddings)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def extract(self, volume):
        """
        Extracts features for a single patient volume.

        Args:
            volume (np.array): Shape (n_slices, H, W, 3), values in [0, 1].

        Returns:
            np.array: Shape (1, embed_dim) - Aggregated feature vector.
        """
        # Ensure input is 4D
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]

        # Convert to Tensor: (N, H, W, C) -> (N, C, H, W)
        # process_patient_scan returns (N, 256, 256, 3)
        tensor = torch.from_numpy(volume).permute(0, 3, 1, 2).float().to(self.device)

        # Normalize
        tensor = self.normalize(tensor)

        with torch.no_grad():
            # Forward pass -> (N, Embed_Dim)
            # For EfficientNet-B0, Embed_Dim is typically 1280
            features = self.model(tensor)

        # Global Average Pooling across slices
        # (N, Embed_Dim) -> (1, Embed_Dim)
        patient_embedding = torch.mean(features, dim=0, keepdim=True)

        return patient_embedding.cpu().numpy()


class PCA_Reducer:
    """
    Reduces dimensionality of extracted features using PCA.
    """

    def __init__(self, n_components=Config.PCA_COMPONENTS):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=Config.SEED)

    def fit(self, X):
        """Fits PCA on the provided data."""
        self.pca.fit(X)

    def transform(self, X):
        """Transforms data using the fitted PCA."""
        return self.pca.transform(X)

    def fit_transform(self, X):
        """Fits and transforms data."""
        return self.pca.fit_transform(X)


def extract_features(df, mode="train", load_cached_data=True):
    """
    Generates or loads deep learning features for a set of patients.

    Args:
        df (pd.DataFrame): Dataframe containing 'Patient' and 'dcm_path'.
        mode (str): 'train', 'val', or 'test' - used for cache naming.
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        np.array: Array of extracted features (N_samples, Embed_Dim).
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"raw_features_{mode}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached raw features for {mode} from {cache_path}")
        return np.load(cache_path)

    # 2. Compute from scratch
    print(f"Extracting CNN features for {mode} set ({len(df)} patients)...")

    extractor = CNNFeatureExtractor()
    features_list = []

    # Iterate over patients
    # Note: We do not use tqdm as per instructions to minimize output
    for _, row in df.iterrows():
        patient_id = row["Patient"]
        rel_path = row["dcm_path"]

        # Construct full path to DICOM directory
        # Metadata dcm_path is relative to input/ (e.g., "train/ID...")
        full_dcm_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load and process scan
        # Returns (Slices, H, W, 3)
        volume = process_patient_scan(
            patient_id=patient_id,
            dcm_path=full_dcm_path,
            load_cached_data=load_cached_data,
            n_slices=Config.SLICES_PER_PATIENT,
            img_size=Config.IMG_SIZE,
        )

        # Extract embeddings
        # Returns (1, Embed_Dim)
        emb = extractor.extract(volume)
        features_list.append(emb)

    # Stack all patient features
    if features_list:
        all_features = np.vstack(features_list)
    else:
        # Fallback for empty dataframe
        all_features = np.zeros((0, 1280))  # Assuming EfficientNet-B0 dim

    # 3. Save to cache
    print(f"Saving {mode} features to {cache_path}")
    np.save(cache_path, all_features)

    return all_features
