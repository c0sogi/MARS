import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from library.config import Config
from library.dicom_processing import process_patient


class VisualFeatureExtractor:
    """
    Extracts deep visual features from patient CT scans using EfficientNet-B0.
    """

    def __init__(self):
        self.device = Config.DEVICE

        # Load pre-trained EfficientNet-B0
        # We use the default IMAGENET1K_V1 weights
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace the classifier head with Identity to extract features
        # Structure: features -> avgpool -> flatten -> classifier
        # By replacing classifier with Identity, we get the output of the flatten layer (1280 dim)
        self.model.classifier = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def extract_single_patient(self, images):
        """
        Runs inference on a single patient's set of images.

        Args:
            images (np.ndarray): Array of shape (6, 224, 224) with values in [0, 1].

        Returns:
            np.ndarray: Flattened feature vector of shape (7680,).
        """
        # Convert numpy array to tensor: (N, H, W) -> (N, 1, H, W)
        tensor = torch.from_numpy(images).unsqueeze(1)

        # Replicate grayscale channel to 3 channels: (N, 3, H, W)
        tensor = tensor.repeat(1, 3, 1, 1)

        # Move to device
        tensor = tensor.to(self.device)

        # Apply normalization
        tensor = self.normalize(tensor)

        with torch.no_grad():
            # Forward pass
            # Output shape: (6, 1280)
            features = self.model(tensor)

        # Move to CPU and flatten
        # We concatenate features from all 6 images (MIPs + Slices)
        features_flat = features.cpu().numpy().flatten()

        return features_flat

    def generate_features(self, df, split_name, load_cached_data=True):
        """
        Generates feature matrix for the provided dataframe.

        Args:
            df (pd.DataFrame): Metadata dataframe containing 'Patient' and 'dcm_path'.
            split_name (str): Name of the split (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Feature matrix of shape (len(df), 7680).
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"cnn_features_{split_name}.npy")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached CNN features for {split_name} from {cache_path}...")
            features = np.load(cache_path)
            # Simple check to ensure length matches (robustness)
            if len(features) == len(df):
                return features
            else:
                print(
                    f"Cached features length ({len(features)}) mismatch with DataFrame ({len(df)}). Recomputing..."
                )

        print(f"Generating CNN features for {split_name}...")

        # 2. Compute features
        # We compute unique patient features first to avoid redundant inference
        # since a patient has multiple rows (weeks) in the dataframe.
        unique_patients = df["Patient"].unique()
        patient_feature_map = {}

        for pid in unique_patients:
            # Get the relative path for this patient
            # We assume the path is consistent for all rows of the same patient
            dcm_path = df[df["Patient"] == pid].iloc[0]["dcm_path"]

            # Load processed images (handles its own caching)
            images = process_patient(pid, dcm_path, load_cached_data=load_cached_data)

            # Extract features via CNN
            feats = self.extract_single_patient(images)
            patient_feature_map[pid] = feats

        # 3. Construct the full matrix aligned with the DataFrame
        all_features = []
        for _, row in df.iterrows():
            pid = row["Patient"]
            all_features.append(patient_feature_map[pid])

        all_features = np.array(all_features, dtype=np.float32)

        # 4. Save to cache
        np.save(cache_path, all_features)
        print(f"Saved CNN features to {cache_path}. Shape: {all_features.shape}")

        return all_features
