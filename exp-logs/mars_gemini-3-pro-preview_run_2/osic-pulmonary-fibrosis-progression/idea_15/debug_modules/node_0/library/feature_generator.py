import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

from library.config import CACHE_DIR, SEED, IMAGE_SIZE
from library.utils import seed_everything
from library.dicom_processor import DicomProcessor


class FeatureGenerator:
    """
    Generates a dense feature vector for a patient by combining:
    1. EfficientNet-B0 features from Multi-Axis CT slices.
    2. Volumetric Density Histogram.
    3. Clinical Metadata (Age, Sex, Smoking, Baseline FVC/Percent).
    """

    def __init__(self):
        seed_everything(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dicom_processor = DicomProcessor()

        # Image Normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self._init_model()

    def _init_model(self):
        """Initializes EfficientNet-B0 and removes the classification head."""
        # Use pretrained weights
        try:
            from torchvision.models import EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b0(weights=weights)
        except ImportError:
            # Fallback for older torchvision versions
            self.model = models.efficientnet_b0(pretrained=True)

        # Replace classifier with Identity to get the 1280-dim feature vector
        # Note: efficientnet_b0 structure: features -> avgpool -> classifier
        # We want the output of avgpool (flattened)
        self.model.classifier = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

    def extract_image_features(self, images):
        """
        Extracts deep features from a batch of images.
        Args:
            images (np.ndarray): Shape (N, H, W), values in [0, 1].
        Returns:
            np.ndarray: Flattened feature vector of shape (N * 1280,).
        """
        # Prepare tensor
        # (N, H, W) -> (N, 1, H, W)
        tensor = torch.from_numpy(images).float().unsqueeze(1)

        # Repeat to 3 channels for ImageNet-pretrained model
        # (N, 1, H, W) -> (N, 3, H, W)
        tensor = tensor.repeat(1, 3, 1, 1)

        # Normalize
        tensor = self.normalize(tensor)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            # Forward pass
            # Output shape: (N, 1280)
            features = self.model(tensor)

        return features.cpu().numpy().flatten()

    def encode_clinical_features(self, clinical_data):
        """
        Encodes clinical metadata into a numerical vector.
        Args:
            clinical_data (dict or pd.Series): Contains Age, Sex, SmokingStatus, etc.
        Returns:
            np.ndarray: Shape (7,).
        """
        # 1. Age
        age = float(clinical_data["Age"])

        # 2. Sex (Male=1, Female=0)
        sex = 1.0 if clinical_data["Sex"] == "Male" else 0.0

        # 3. SmokingStatus (One-Hot)
        # Categories: Ex-smoker, Never smoked, Currently smokes
        status = clinical_data["SmokingStatus"]
        smoking_ohe = [0.0, 0.0, 0.0]
        if status == "Ex-smoker":
            smoking_ohe[0] = 1.0
        elif status == "Never smoked":
            smoking_ohe[1] = 1.0
        elif status == "Currently smokes":
            smoking_ohe[2] = 1.0

        # 4. Baseline FVC and Percent
        # Try 'Baseline_FVC' (test set format), fallback to 'FVC' (train set format)
        fvc = float(clinical_data.get("Baseline_FVC", clinical_data.get("FVC", 0)))
        percent = float(
            clinical_data.get("Baseline_Percent", clinical_data.get("Percent", 0))
        )

        return np.array([age, sex] + smoking_ohe + [fvc, percent], dtype=np.float32)

    def generate_patient_features(
        self, patient_id, dcm_path, clinical_data, load_cached_data=True
    ):
        """
        Generates the full feature vector for a patient.
        Checks cache first.

        Args:
            patient_id (str): Unique Patient ID.
            dcm_path (str): Path to DICOM directory.
            clinical_data (dict/Series): Clinical metadata.
            load_cached_data (bool): Whether to use cached .npy files.

        Returns:
            np.ndarray: Concatenated feature vector.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{patient_id}_features.npy")

        # 1. Load from Cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except Exception:
                pass  # Proceed to generate if load fails

        # 2. Generate Features
        # a. Image Processing (Texture + Histogram)
        images, histogram = self.dicom_processor.process_patient(
            patient_id, dcm_path, load_cached_data=load_cached_data
        )

        # b. Deep Feature Extraction
        texture_features = self.extract_image_features(images)

        # c. Clinical Encoding
        clinical_features = self.encode_clinical_features(clinical_data)

        # d. Concatenate
        # Texture (~7680) + Histogram (4) + Clinical (7)
        full_features = np.concatenate([texture_features, histogram, clinical_features])

        # 3. Save to Cache
        np.save(cache_file, full_features)

        return full_features

    def process_dataset(self, df, load_cached_data=True):
        """
        Helper to process a dataframe of patients.
        Returns a dictionary: {PatientID: FeatureVector}
        """
        features_map = {}
        unique_patients = df["Patient"].unique()

        for pid in unique_patients:
            # Extract the first row for this patient to get static metadata
            # (Assumes caller has handled baseline logic or all rows have static data)
            row = df[df["Patient"] == pid].iloc[0]
            dcm_path = row["dcm_path"]

            feats = self.generate_patient_features(pid, dcm_path, row, load_cached_data)
            features_map[pid] = feats

        return features_map
