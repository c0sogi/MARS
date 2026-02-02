import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import cv2
from library.config import Config
from library.utils import seed_everything

# Try importing pydicom, handle case if not available (though standard for this task)
try:
    import pydicom
except ImportError:
    pydicom = None


class VarianceSelector:
    """
    Selects the top N slices based on pixel variance to capture tissue heterogeneity.
    Handles DICOM loading, windowing, and preprocessing.
    """

    def __init__(self, n_slices=Config.N_SLICES, img_size=Config.IMG_SIZE):
        self.n_slices = n_slices
        self.img_size = img_size
        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _read_dicom(self, path):
        """
        Reads a DICOM file and returns the pixel array in Hounsfield Units.
        """
        # Priority 1: pydicom
        if pydicom:
            try:
                dcm = pydicom.dcmread(path)
                image = dcm.pixel_array.astype(np.float32)

                # Apply Rescale Slope/Intercept to get HU
                if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                    slope = float(dcm.RescaleSlope)
                    intercept = float(dcm.RescaleIntercept)
                    image = image * slope + intercept
                return image
            except Exception:
                pass

        # Priority 2: OpenCV (fallback)
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                return image.astype(np.float32)
        except Exception:
            pass

        return None

    def _preprocess(self, image):
        """
        Applies Lung Windowing, Resizing, and Normalization.
        """
        # Lung Windowing: Level -600, Width 1500 -> Range [-1350, 150]
        L, W = -600, 1500
        lower, upper = L - W // 2, L + W // 2

        image = np.clip(image, lower, upper)
        # Normalize to [0, 1]
        image = (image - lower) / (upper - lower)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Stack to 3 channels (Grayscale -> RGB)
        image = np.stack([image, image, image], axis=-1)

        # ImageNet Standardization
        image = (image - self.mean) / self.std

        # HWC to CHW
        image = image.transpose(2, 0, 1)
        return image

    def select_slices(self, dcm_dir):
        """
        Loads all slices, computes variance, and returns the top N preprocessed slices.
        Returns tensor of shape (N_SLICES, 3, H, W).
        """
        full_path = os.path.join(Config.INPUT_DIR, dcm_dir)

        # Handle missing directory
        if not os.path.exists(full_path):
            return np.zeros(
                (self.n_slices, 3, self.img_size, self.img_size), dtype=np.float32
            )

        # List files
        files = [
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if f.lower().endswith(".dcm")
        ]
        if not files:
            return np.zeros(
                (self.n_slices, 3, self.img_size, self.img_size), dtype=np.float32
            )

        slice_candidates = []

        for f in files:
            img = self._read_dicom(f)
            if img is not None:
                # Calculate variance on the HU values (before windowing/clipping)
                # This better captures the texture difference between lung and air/bone
                var = np.var(img)
                slice_candidates.append((var, img))

        if not slice_candidates:
            return np.zeros(
                (self.n_slices, 3, self.img_size, self.img_size), dtype=np.float32
            )

        # Sort by variance descending
        slice_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select Top N
        top_slices = slice_candidates[: self.n_slices]

        # Preprocess selected slices
        processed_batch = []
        for _, img in top_slices:
            processed_batch.append(self._preprocess(img))

        processed_batch = np.array(processed_batch)

        # Pad with zeros if fewer than N slices found
        if len(processed_batch) < self.n_slices:
            pad_len = self.n_slices - len(processed_batch)
            padding = np.zeros(
                (pad_len, 3, self.img_size, self.img_size), dtype=np.float32
            )
            processed_batch = np.concatenate([processed_batch, padding], axis=0)

        return processed_batch


class DeepFeatureExtractor:
    """
    Wraps EfficientNet-B0 to extract deep features.
    """

    def __init__(self, device):
        self.device = device
        # Create EfficientNet-B0, num_classes=0 removes classifier, returns pooled features
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.model.to(device)
        self.model.eval()

    def extract(self, img_tensor):
        """
        Args:
            img_tensor: (Batch, C, H, W)
        Returns:
            features: (Batch, Feature_Dim)
        """
        with torch.no_grad():
            features = self.model(img_tensor)
        return features


class ImageExtractor:
    """
    Main class for the Heterogeneity-Aware Feature Extraction pipeline.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.selector = VarianceSelector()
        self.extractor = DeepFeatureExtractor(device)

    def _dual_moment_pooling(self, features):
        """
        Aggregates slice features using Mean and Standard Deviation.
        Args:
            features: Tensor of shape (N_SLICES, Feature_Dim)
        Returns:
            numpy array of shape (2 * Feature_Dim,)
        """
        # Mean embedding
        mean_feat = torch.mean(features, dim=0)

        # Std embedding (captures heterogeneity)
        std_feat = torch.std(features, dim=0)

        # Handle NaN if N=1 (std is undefined/0)
        if torch.isnan(std_feat).any():
            std_feat = torch.nan_to_num(std_feat)

        # Concatenate
        combined = torch.cat([mean_feat, std_feat], dim=0)
        return combined.cpu().numpy()

    def extract_features(self, metadata_df, load_cached_data=True):
        """
        Extracts features for all unique patients in the metadata DataFrame.
        Handles caching to ./working/idea_17/

        Args:
            metadata_df: DataFrame containing 'Patient' and 'dcm_path' columns.
            load_cached_data: If True, attempts to load from disk.

        Returns:
            dict: {PatientID: FeatureVector (np.array)}
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        features_map = {}

        # Get unique patients to process
        unique_patients = metadata_df[["Patient", "dcm_path"]].drop_duplicates()

        print(f"Extracting features for {len(unique_patients)} patients...")

        for _, row in unique_patients.iterrows():
            patient_id = row["Patient"]
            dcm_path = row["dcm_path"]

            cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

            # 1. Check Cache
            if load_cached_data and os.path.exists(cache_path):
                try:
                    feat = np.load(cache_path)
                    features_map[patient_id] = feat
                    continue
                except Exception:
                    # If load fails, recompute
                    pass

            # 2. Select Slices (Variance Based)
            # Returns (N, 3, H, W)
            img_batch_np = self.selector.select_slices(dcm_path)

            # 3. Extract Deep Features
            img_tensor = torch.tensor(img_batch_np).float().to(self.device)
            # Returns (N, 1280)
            slice_features = self.extractor.extract(img_tensor)

            # 4. Dual Moment Pooling
            # Returns (2560,)
            patient_feature = self._dual_moment_pooling(slice_features)

            # 5. Save to Cache
            np.save(cache_path, patient_feature)
            features_map[patient_id] = patient_feature

        return features_map
