import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import Config


class FusionPipeline:
    """
    Manages the preprocessing, dimensionality reduction, and fusion of
    tabular and image features.

    Persists fitted state (means, scales, components) to disk as .npy files
    to avoid re-fitting and ensure consistent transformation across runs.
    """

    def __init__(self):
        # Tabular Scaler State
        self.scaler_mean = None
        self.scaler_scale = None

        # CNN PCA State
        self.cnn_pca_mean = None
        self.cnn_pca_components = None

        # ViT PCA State
        self.vit_pca_mean = None
        self.vit_pca_components = None

    def fit(self, train_data, load_cached_data=True):
        """
        Fits the StandardScaler and PCA models on the training data.

        Args:
            train_data (dict): Dictionary containing 'tab', 'cnn', 'vit' numpy arrays.
            load_cached_data (bool): If True, attempts to load fitted state from cache.
        """
        Config.setup()

        # Define cache paths
        path_scaler_mean = Config.get_cache_path(Config.CACHE_SCALER_MEAN)
        path_scaler_scale = Config.get_cache_path(Config.CACHE_SCALER_SCALE)

        path_cnn_mean = Config.get_cache_path(Config.CACHE_PCA_CNN_MEAN)
        path_cnn_comps = Config.get_cache_path(Config.CACHE_PCA_CNN_COMPONENTS)

        path_vit_mean = Config.get_cache_path(Config.CACHE_PCA_VIT_MEAN)
        path_vit_comps = Config.get_cache_path(Config.CACHE_PCA_VIT_COMPONENTS)

        paths = [
            path_scaler_mean,
            path_scaler_scale,
            path_cnn_mean,
            path_cnn_comps,
            path_vit_mean,
            path_vit_comps,
        ]

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(p) for p in paths):
            print("Loading preprocessing state from cache...")
            self.scaler_mean = np.load(path_scaler_mean)
            self.scaler_scale = np.load(path_scaler_scale)

            self.cnn_pca_mean = np.load(path_cnn_mean)
            self.cnn_pca_components = np.load(path_cnn_comps)

            self.vit_pca_mean = np.load(path_vit_mean)
            self.vit_pca_components = np.load(path_vit_comps)
            return

        # 2. Fit Fresh Models
        print("Fitting preprocessing pipeline...")

        # A. Tabular Features - Standard Scaling
        print("  Fitting StandardScaler on tabular features...")
        scaler = StandardScaler()
        scaler.fit(train_data["tab"])
        self.scaler_mean = scaler.mean_
        self.scaler_scale = scaler.scale_

        # B. CNN Embeddings - PCA
        print(f"  Fitting PCA on CNN embeddings (Variance={Config.PCA_VARIANCE})...")
        cnn_pca = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
        cnn_pca.fit(train_data["cnn"])
        self.cnn_pca_mean = cnn_pca.mean_
        self.cnn_pca_components = cnn_pca.components_
        print(f"    CNN Components retained: {cnn_pca.n_components_}")

        # C. ViT Embeddings - PCA
        print(f"  Fitting PCA on ViT embeddings (Variance={Config.PCA_VARIANCE})...")
        vit_pca = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
        vit_pca.fit(train_data["vit"])
        self.vit_pca_mean = vit_pca.mean_
        self.vit_pca_components = vit_pca.components_
        print(f"    ViT Components retained: {vit_pca.n_components_}")

        # 3. Save State to Cache
        print("Saving preprocessing state to cache...")
        np.save(path_scaler_mean, self.scaler_mean)
        np.save(path_scaler_scale, self.scaler_scale)

        np.save(path_cnn_mean, self.cnn_pca_mean)
        np.save(path_cnn_comps, self.cnn_pca_components)

        np.save(path_vit_mean, self.vit_pca_mean)
        np.save(path_vit_comps, self.vit_pca_components)

    def transform(self, data_dict):
        """
        Applies the learned transformations to the data and fuses features.

        Args:
            data_dict (dict): Dictionary containing 'tab', 'cnn', 'vit' numpy arrays.

        Returns:
            np.ndarray: Fused feature matrix.
        """
        if self.scaler_mean is None:
            raise RuntimeError("Pipeline must be fitted before transform.")

        # 1. Transform Tabular Features (Standardization)
        # Formula: z = (x - u) / s
        tab = data_dict["tab"]
        tab_scaled = (tab - self.scaler_mean) / self.scaler_scale

        # 2. Transform CNN Embeddings (PCA)
        # Formula: T = (X - mean) @ components.T
        cnn = data_dict["cnn"]
        cnn_centered = cnn - self.cnn_pca_mean
        cnn_pca = np.dot(cnn_centered, self.cnn_pca_components.T)

        # 3. Transform ViT Embeddings (PCA)
        # Formula: T = (X - mean) @ components.T
        vit = data_dict["vit"]
        vit_centered = vit - self.vit_pca_mean
        vit_pca = np.dot(vit_centered, self.vit_pca_components.T)

        # 4. Feature Fusion (Concatenation)
        fused = np.hstack([tab_scaled, cnn_pca, vit_pca])

        return fused.astype(np.float32)
