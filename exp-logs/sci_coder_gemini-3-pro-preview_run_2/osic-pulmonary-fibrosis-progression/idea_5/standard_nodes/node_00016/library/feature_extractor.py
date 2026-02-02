import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from library.config import Config
from library.image_loader import get_patient_images


class VisualEncoder:
    """
    Deep learning feature extractor based on EfficientNet-B0.
    Extracts high-level texture and structural features from CT slices.
    """

    def __init__(self):
        self.device = Config.DEVICE
        # Load EfficientNet-B0 with default ImageNet weights
        # We use a try-except block to handle potential version differences in torchvision
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions or different environments
            self.model = models.efficientnet_b0(pretrained=True)

        # Remove the classification head (fc) to output raw features
        # EfficientNet-B0 features before classifier are 1280-dimensional
        self.model.classifier = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, images):
        """
        Extract features from a batch of images.

        Args:
            images (torch.Tensor): Input tensor of shape (Batch, 3, H, W).

        Returns:
            np.ndarray: Feature vectors of shape (Batch, Feature_Dim).
        """
        with torch.no_grad():
            images = images.to(self.device)
            features = self.model(images)
            return features.cpu().numpy()


def extract_patient_embedding(encoder, patient_id, dcm_path_rel):
    """
    Generates a single embedding vector for a patient by averaging features
    across selected CT slices.

    Args:
        encoder (VisualEncoder): The initialized neural network encoder.
        patient_id (str): The patient ID.
        dcm_path_rel (str): Relative path to the patient's DICOM folder.

    Returns:
        np.ndarray: A 1D numpy array of shape (Feature_Dim,).
    """
    # Load preprocessed tensor: (NUM_SLICES, 3, IMG_SIZE, IMG_SIZE)
    # This function internally handles caching of the processed image tensor
    img_tensor = get_patient_images(patient_id, dcm_path_rel)

    # Extract features: (NUM_SLICES, 1280)
    slice_features = encoder(img_tensor)

    # Global Average Pooling across slices
    # We average the feature vectors of the 5 selected slices to get one patient representation
    patient_embedding = np.mean(slice_features, axis=0)

    return patient_embedding


def generate_embeddings(
    metadata_df, cache_filename, load_cached_data=Config.LOAD_CACHED_DATA
):
    """
    Generates feature embeddings for all samples in the metadata DataFrame.
    Includes caching logic to save/load the full feature matrix.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'Patient' and 'dcm_path' columns.
        cache_filename (str): Name of the cache file (e.g., 'train_features.npy').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Feature matrix of shape (len(metadata_df), Feature_Dim).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        try:
            embeddings = np.load(cache_path)
            if len(embeddings) == len(metadata_df):
                return embeddings
            else:
                print(
                    f"Cache dimension mismatch (Found {len(embeddings)}, Expected {len(metadata_df)}). Recomputing."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing.")

    # 2. Compute Embeddings
    print(f"Computing embeddings for {len(metadata_df)} samples...")
    encoder = VisualEncoder()

    # Optimization: The dataframe may have multiple rows per patient (longitudinal data),
    # but the image embedding is static per patient. We compute unique patients first.
    unique_patients_df = metadata_df[["Patient", "dcm_path"]].drop_duplicates()
    patient_embedding_map = {}

    # Process unique patients
    for _, row in unique_patients_df.iterrows():
        pid = row["Patient"]
        path = row["dcm_path"]
        embedding = extract_patient_embedding(encoder, pid, path)
        patient_embedding_map[pid] = embedding

    # Map back to the original dataframe order
    embeddings_list = []
    for pid in metadata_df["Patient"]:
        embeddings_list.append(patient_embedding_map[pid])

    embeddings = np.array(embeddings_list)

    # 3. Save Cache
    try:
        np.save(cache_path, embeddings)
        print(f"Saved embeddings to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return embeddings
