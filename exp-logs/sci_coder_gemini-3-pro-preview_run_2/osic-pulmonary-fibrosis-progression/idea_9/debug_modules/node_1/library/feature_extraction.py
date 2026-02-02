import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from library.config import Config
from library.image_processing import process_patient

# ==========================================
# Reproducibility
# ==========================================
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class TextureExtractor:
    """
    Wraps EfficientNet-B0 to extract texture features from lung slices.
    Uses a pre-trained backbone to generate embeddings for the stratified slices.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Load Pre-trained EfficientNet-B0 (ImageNet weights)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        base_model = models.efficientnet_b0(weights=weights)

        # Isolate the feature extractor:
        # Structure: features (Conv) -> avgpool (AdaptiveAvgPool) -> classifier (Linear)
        # We discard the classifier to get the 1280-dim embedding.
        self.feature_extractor = nn.Sequential(
            base_model.features, base_model.avgpool, nn.Flatten()
        )

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Standard ImageNet normalization parameters
        # Shape: (1, 3, 1, 1) for broadcasting over (N, C, H, W)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
            1, 3, 1, 1
        )

    def extract(self, slices):
        """
        Extracts features from a set of slices.

        Args:
            slices (np.ndarray): Shape (3, 224, 224), values in [0, 1].

        Returns:
            np.ndarray: Flattened feature vector of shape (3840,).
                        (3 slices * 1280 features per slice)
        """
        # Convert input to tensor
        # Input is (3, H, W) -> need (3, C, H, W) where C=3 for RGB
        x = torch.tensor(slices, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(1)  # (3, 1, H, W)
        x = x.repeat(1, 3, 1, 1)  # (3, 3, H, W)

        # Apply Normalization
        x = (x - self.mean) / self.std

        with torch.no_grad():
            # Extract features: Output shape (3, 1280)
            features = self.feature_extractor(x)

        # Flatten all slices into a single vector for the patient
        return features.cpu().numpy().flatten()


class FeaturePipeline:
    """
    Manages the extraction of texture and histogram features,
    and the application of PCA for dimensionality reduction.
    """

    def __init__(self):
        self.pca_params_path = os.path.join(Config.WORKING_DIR, "pca_params.npz")
        self.extractor = TextureExtractor()

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def _get_cache_paths(self, mode):
        """
        Returns paths for raw and final feature caches.
        Appends debug suffix if running on a subset to prevent cache collisions.
        """
        suffix = ""
        if Config.DEBUG_DATA_SIZE is not None:
            suffix = f"_debug_{Config.DEBUG_DATA_SIZE}"

        raw_tex = os.path.join(Config.WORKING_DIR, f"raw_tex_{mode}{suffix}.npy")
        raw_hist = os.path.join(Config.WORKING_DIR, f"raw_hist_{mode}{suffix}.npy")
        ids = os.path.join(Config.WORKING_DIR, f"ids_{mode}{suffix}.npy")
        final_X = os.path.join(Config.WORKING_DIR, f"final_X_{mode}{suffix}.npy")

        return raw_tex, raw_hist, ids, final_X

    def process_dataset(self, metadata_df, mode="train", load_cached_data=True):
        """
        Main entry point to process a dataset (train/val/test).

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'Patient' and 'dcm_path'.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            pd.DataFrame: DataFrame with Patient IDs and extracted features.
        """
        raw_tex_path, raw_hist_path, ids_path, final_X_path = self._get_cache_paths(
            mode
        )

        # --- Step 1: Check for Final Processed Data ---
        if (
            load_cached_data
            and os.path.exists(final_X_path)
            and os.path.exists(ids_path)
        ):
            print(f"Loading final processed features for {mode} from cache.")
            X = np.load(final_X_path)
            patient_ids = np.load(ids_path)
            return self._wrap_dataframe(patient_ids, X)

        # --- Step 2: Check for/Generate Raw Features ---
        if (
            load_cached_data
            and os.path.exists(raw_tex_path)
            and os.path.exists(raw_hist_path)
            and os.path.exists(ids_path)
        ):
            print(f"Loading raw features for {mode} from cache.")
            raw_tex = np.load(raw_tex_path)
            raw_hist = np.load(raw_hist_path)
            patient_ids = np.load(ids_path)
        else:
            print(f"Generating raw features for {mode}...")
            raw_tex, raw_hist, patient_ids = self._generate_raw_features(
                metadata_df, load_cached_data
            )

            # Cache raw features
            np.save(raw_tex_path, raw_tex)
            np.save(raw_hist_path, raw_hist)
            np.save(ids_path, patient_ids)

        # --- Step 3: Apply PCA ---
        if mode == "train":
            print("Fitting PCA on training data...")
            # Dynamically adjust n_components based on sample size (Cite debug_lesson_7)
            n_samples = raw_tex.shape[0]
            n_components = min(Config.N_PCA_COMPONENTS, n_samples)

            pca = PCA(n_components=n_components, random_state=Config.SEED)
            pca_tex = pca.fit_transform(raw_tex)

            # Save PCA parameters (Mean and Components) manually to avoid pickle issues
            np.savez(self.pca_params_path, mean=pca.mean_, components=pca.components_)
            print(
                f"PCA Fitted. Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.6f}"
            )

        else:
            print("Applying PCA transform...")
            if not os.path.exists(self.pca_params_path):
                raise FileNotFoundError(
                    "PCA parameters not found. You must process the 'train' set first to fit PCA."
                )

            # Load PCA parameters
            pca_data = np.load(self.pca_params_path)
            mean = pca_data["mean"]
            components = pca_data["components"]

            # Manual Transform: (X - mean) @ components.T
            pca_tex = (raw_tex - mean) @ components.T

        # --- Step 4: Concatenate Features ---
        # Combine PCA-reduced texture features with the Density Histogram
        final_X = np.concatenate([pca_tex, raw_hist], axis=1)

        # Cache final result
        np.save(final_X_path, final_X)
        np.save(
            ids_path, patient_ids
        )  # Ensure IDs are also saved/synced with final cache

        return self._wrap_dataframe(patient_ids, final_X)

    def _generate_raw_features(self, df, load_cached_image_data):
        """Iterates through patients and extracts raw texture/histogram data."""
        unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()

        # Handle Debugging Size
        if Config.DEBUG_DATA_SIZE is not None:
            print(f"Debug Mode: Limiting to {Config.DEBUG_DATA_SIZE} patients.")
            unique_patients = unique_patients.head(Config.DEBUG_DATA_SIZE)

        raw_tex_list = []
        raw_hist_list = []
        patient_ids = []

        for _, row in unique_patients.iterrows():
            pid = row["Patient"]
            dcm_path = row["dcm_path"]

            # Load image data (slices + histogram)
            # process_patient handles its own caching of the heavy image processing
            data = process_patient(
                pid, dcm_path, load_cached_data=load_cached_image_data
            )

            # Extract Deep Texture Features
            tex_vec = self.extractor.extract(data["slices"])

            raw_tex_list.append(tex_vec)
            raw_hist_list.append(data["histogram"])
            patient_ids.append(pid)

        return (
            np.array(raw_tex_list, dtype=np.float32),
            np.array(raw_hist_list, dtype=np.float32),
            np.array(patient_ids),
        )

    def _wrap_dataframe(self, ids, X):
        """Helper to create a DataFrame from features."""
        cols = [f"feat_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["Patient"] = ids
        return df
