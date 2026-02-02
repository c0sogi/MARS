import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from library.config import Config
from library.dicom_handler import DicomHandler
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles deep feature extraction using EfficientNet-B0 and Dual-Moment Aggregation.
    """

    def __init__(self):
        """
        Initializes the model and device.
        """
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # Load Pretrained EfficientNet-B0
        # We use the default weights (IMAGENET1K_V1)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)

        # We only need the feature extractor part, not the classifier
        # EfficientNet structure: features -> avgpool -> classifier
        # We will manually forward pass through features and avgpool
        self.model.to(self.device)
        self.model.eval()

    def _extract_batch(self, images_tensor):
        """
        Extracts features for a batch of images (slices).

        Args:
            images_tensor (torch.Tensor): Shape (N, 3, H, W)

        Returns:
            torch.Tensor: Shape (N, 1280)
        """
        with torch.no_grad():
            images_tensor = images_tensor.to(self.device)

            # Forward pass through feature extractor
            # Output: (N, 1280, 7, 7) for 224x224 input
            x = self.model.features(images_tensor)

            # Global Average Pooling
            # Output: (N, 1280, 1, 1)
            x = self.model.avgpool(x)

            # Flatten
            # Output: (N, 1280)
            x = torch.flatten(x, 1)

        return x

    def _aggregate_dual_moment(self, features):
        """
        Computes Mean and Std across the slice dimension.

        Args:
            features (torch.Tensor): Shape (N_SLICES, 1280)

        Returns:
            np.ndarray: Concatenated Mean+Std vector of shape (2560,)
        """
        # Compute Mean and Std across slices (dim 0)
        # If N_SLICES=1, std will be NaN or 0. Handled by torch.std (unbiased=True by default)
        # However, if N=1, std is 0.

        if features.size(0) > 1:
            mean_feat = torch.mean(features, dim=0)
            std_feat = torch.std(features, dim=0)
        else:
            mean_feat = features[0]
            std_feat = torch.zeros_like(mean_feat)

        # Concatenate
        combined = torch.cat([mean_feat, std_feat], dim=0)

        return combined.cpu().numpy()

    def process_dataset(self, subset, load_cached_data=True):
        """
        Processes all patients in the specified subset, extracting and aggregating features.
        Handles caching to disk.

        Args:
            subset (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            tuple: (features_array, patient_ids_array)
                features_array: (N_Patients, 2560)
                patient_ids_array: (N_Patients,)
        """
        # Define cache paths
        feature_cache_path = os.path.join(Config.CACHE_DIR, f"features_{subset}.npy")
        ids_cache_path = os.path.join(Config.CACHE_DIR, f"ids_{subset}.npy")

        # 1. Try loading from cache
        if load_cached_data:
            if os.path.exists(feature_cache_path) and os.path.exists(ids_cache_path):
                print(
                    f"Loading cached features for {subset} from {Config.CACHE_DIR}..."
                )
                try:
                    features = np.load(feature_cache_path)
                    ids = np.load(ids_cache_path)
                    return features, ids
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Determine Metadata Path
        if subset == "train":
            meta_path = Config.TRAIN_META_PATH
        elif subset == "val":
            meta_path = Config.VAL_META_PATH
        elif subset == "test":
            meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown subset: {subset}")

        # 3. Load Metadata and get Unique Patients
        df = pd.read_csv(meta_path)
        unique_patients = df["Patient"].unique()

        # Debugging: Limit sample size
        if Config.DEBUG:
            unique_patients = unique_patients[: Config.DEBUG_SAMPLE_SIZE]
            print(f"DEBUG MODE: Processing only {len(unique_patients)} patients.")

        print(
            f"Extracting features for {len(unique_patients)} patients in {subset} set..."
        )

        features_list = []
        ids_list = []

        # 4. Processing Loop
        for patient_id in unique_patients:
            # Load and preprocess images (N_SLICES, 3, H, W)
            # DicomHandler handles the caching of image arrays
            img_array = DicomHandler.process_patient(
                patient_id, subset=subset, load_cached_data=load_cached_data
            )

            # Convert to Tensor
            img_tensor = torch.from_numpy(img_array)

            # Extract features (N_SLICES, 1280)
            slice_features = self._extract_batch(img_tensor)

            # Aggregate (2560,)
            patient_features = self._aggregate_dual_moment(slice_features)

            features_list.append(patient_features)
            ids_list.append(patient_id)

        # 5. Convert to Arrays
        features_array = np.array(features_list, dtype=np.float32)
        ids_array = np.array(ids_list)

        # 6. Save to Cache
        try:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            np.save(feature_cache_path, features_array)
            np.save(ids_cache_path, ids_array)
            print(f"Saved features to {feature_cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

        return features_array, ids_array
