import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from sklearn.decomposition import PCA
from library.config import Config
from library.image_utils import select_slices, load_and_preprocess_scan


class CNNEncoder:
    """
    Wraps a pre-trained CNN backbone to extract features from CT slices.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load Pre-trained ResNet18
        # Using the modern 'weights' parameter if available, else fallback
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.model = models.resnet18(pretrained=True)

        # Remove the fully connected layer to get feature embeddings
        # ResNet18 structure: ... -> avgpool -> fc
        # We keep everything up to avgpool
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])

        self.backbone.to(self.device)
        self.backbone.eval()

    def extract_features(self, dcm_dir):
        """
        Extracts features for a single patient (directory of DICOMs).

        Args:
            dcm_dir (str): Path to the patient's DICOM directory.

        Returns:
            np.ndarray: A 1D numpy array of shape (512,) containing the averaged feature vector.
        """
        # 1. Select representative slices
        slice_paths = select_slices(dcm_dir)
        if not slice_paths:
            return np.zeros(512, dtype=np.float32)

        # 2. Preprocess images into a batch tensor
        # Shape: (Batch_Size, 3, H, W)
        img_tensor = load_and_preprocess_scan(slice_paths)
        if img_tensor.shape[0] == 0:
            return np.zeros(512, dtype=np.float32)

        img_tensor = img_tensor.to(self.device)

        # 3. Inference
        with torch.no_grad():
            # Forward pass
            # Output shape: (Batch_Size, 512, 1, 1)
            features = self.backbone(img_tensor)

            # Flatten to (Batch_Size, 512)
            features = features.view(features.size(0), -1)

        # 4. Aggregation
        # Move to CPU and numpy
        features_np = features.cpu().numpy()

        # Average pooling across the selected slices to get a single patient descriptor
        patient_embedding = np.mean(features_np, axis=0)

        return patient_embedding.astype(np.float32)


class FeatureReducer:
    """
    Manages dimensionality reduction using PCA.
    """

    def __init__(self, n_components=None):
        self.n_components = (
            n_components if n_components is not None else Config.N_PCA_COMPONENTS
        )
        self.pca = PCA(n_components=self.n_components, random_state=Config.SEED)
        self.is_fitted = False

    def fit(self, X):
        """
        Fit PCA on the provided data.
        """
        self.pca.fit(X)
        self.is_fitted = True

    def transform(self, X):
        """
        Apply PCA transformation.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureReducer must be fitted before calling transform."
            )
        return self.pca.transform(X)

    def save_state(self, path):
        """
        Saves the PCA components and mean to a .npz file.
        """
        np.savez(
            path,
            components=self.pca.components_,
            mean=self.pca.mean_,
            explained_variance=self.pca.explained_variance_,
        )

    def load_state(self, path):
        """
        Loads PCA state from a .npz file.
        """
        data = np.load(path)
        self.pca.components_ = data["components"]
        self.pca.mean_ = data["mean"]
        self.pca.explained_variance_ = data["explained_variance"]
        self.is_fitted = True


def _get_raw_features_for_split(df, split_name, encoder, load_cached_data):
    """
    Internal helper to process a dataframe of patients and return raw CNN features.
    Handles caching using .npy for features and .parquet for IDs.
    """
    cache_features_path = os.path.join(
        Config.CACHE_DIR, f"raw_features_{split_name}.npy"
    )
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"patient_ids_{split_name}.parquet")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(cache_features_path)
        and os.path.exists(cache_ids_path)
    ):
        print(f"Loading cached raw features for {split_name}...")
        features = np.load(cache_features_path)
        ids_df = pd.read_parquet(cache_ids_path)
        ids = ids_df["Patient"].values
        return ids, features

    print(f"Generating raw features for {split_name}...")

    # Get unique patients and their dcm paths
    # The metadata contains multiple rows per patient (history), we only need unique patients
    unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()

    patient_ids = []
    feature_list = []

    # Iterate over patients
    for _, row in unique_patients.iterrows():
        pid = row["Patient"]
        rel_path = row["dcm_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Extract features using CNN
        emb = encoder.extract_features(full_path)

        patient_ids.append(pid)
        feature_list.append(emb)

    # Convert to arrays
    ids_arr = np.array(patient_ids)
    features_arr = np.array(feature_list, dtype=np.float32)

    # Save cache
    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    np.save(cache_features_path, features_arr)
    pd.DataFrame({"Patient": ids_arr}).to_parquet(cache_ids_path, index=False)

    return ids_arr, features_arr


def generate_features(load_cached_data=True, debug=Config.DEBUG):
    """
    Main pipeline function to generate processed features.

    Args:
        load_cached_data (bool): If True, attempts to load raw features from disk.
        debug (bool): If True, processes only a small subset of data.

    Returns:
        tuple: (train_feat_dict, val_feat_dict, test_feat_dict)
               Each is a dictionary mapping PatientID (str) -> Feature Vector (np.array).
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Handle Debug Mode
    if debug:
        print("DEBUG Mode: Limiting dataset size...")
        train_patients = train_df["Patient"].unique()[:5]
        val_patients = val_df["Patient"].unique()[:5]
        test_patients = test_df["Patient"].unique()[:5]

        train_df = train_df[train_df["Patient"].isin(train_patients)]
        val_df = val_df[val_df["Patient"].isin(val_patients)]
        test_df = test_df[test_df["Patient"].isin(test_patients)]

        # Disable loading cache in debug mode to avoid cache pollution/mismatch
        load_cached_data = False

    # 2. Initialize CNN Encoder
    encoder = CNNEncoder()

    # 3. Extract Raw Features (with caching)
    train_ids, train_raw = _get_raw_features_for_split(
        train_df, "train", encoder, load_cached_data
    )
    val_ids, val_raw = _get_raw_features_for_split(
        val_df, "val", encoder, load_cached_data
    )
    test_ids, test_raw = _get_raw_features_for_split(
        test_df, "test", encoder, load_cached_data
    )

    # 4. PCA Dimensionality Reduction
    print(f"Fitting PCA (n_components={Config.N_PCA_COMPONENTS}) on training data...")
    reducer = FeatureReducer()
    reducer.fit(train_raw)

    # Transform all sets
    train_pca = reducer.transform(train_raw)
    val_pca = reducer.transform(val_raw)
    test_pca = reducer.transform(test_raw)

    # 5. Convert to Dictionaries
    train_feat_dict = dict(zip(train_ids, train_pca))
    val_feat_dict = dict(zip(val_ids, val_pca))
    test_feat_dict = dict(zip(test_ids, test_pca))

    print("Feature generation complete.")
    return train_feat_dict, val_feat_dict, test_feat_dict
