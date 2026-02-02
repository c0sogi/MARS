import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from library.config import Config
from library.feature_extraction import FeatureExtractor
from library.data_loader import load_tabular_data
from library.utils import seed_everything


class FeaturePipeline:
    """
    Manages feature extraction, transformation, fusion, and caching.
    Implements the preprocessing pipeline:
    1. Load/Extract raw features (DINOv2, ConvNeXt, Tabular).
    2. Fit transformers (PCA, QuantileTransformer) on Training data.
    3. Transform and Fuse features.
    4. Cache processed matrices.
    """

    def __init__(self):
        seed_everything(Config.SEED)

        # Transformers
        # PCA for DINOv2 features
        self.pca_dino = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
        # PCA for ConvNeXt features
        self.pca_conv = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
        # QuantileTransformer for Tabular features
        self.qt_tabular = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        self.is_fitted = False
        self.feature_extractor = FeatureExtractor()

    def _ensure_fitted(self, load_cached_data=True):
        """
        Ensures transformers are fitted on training data.
        If not fitted, loads raw training features and fits them.
        """
        if self.is_fitted:
            return

        print("Fitting transformers on training data...")

        # Load raw training features
        # Note: FeatureExtractor handles caching internally
        dino_train, conv_train, _, _ = self.feature_extractor.extract_features(
            split="train", load_cached_data=load_cached_data
        )

        tab_train, _, _ = load_tabular_data(
            split="train", load_cached_data=load_cached_data
        )

        # Fit transformers
        print(f"  Fitting PCA on DINO features (input shape: {dino_train.shape})...")
        self.pca_dino.fit(dino_train)

        print(
            f"  Fitting PCA on ConvNeXt features (input shape: {conv_train.shape})..."
        )
        self.pca_conv.fit(conv_train)

        print(
            f"  Fitting QuantileTransformer on Tabular features (input shape: {tab_train.shape})..."
        )
        self.qt_tabular.fit(tab_train)

        self.is_fitted = True
        print("Transformers fitted successfully.")

    def get_processed_data(self, split, load_cached_data=True):
        """
        Retrieves processed and fused features for a specific split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached fused data.

        Returns:
            tuple: (X_fused, y, ids)
        """
        # Define cache paths for the final fused data
        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        path_X = os.path.join(cache_dir, f"{split}_fused_X.npy")
        path_y = os.path.join(cache_dir, f"{split}_fused_y.npy")
        path_ids = os.path.join(cache_dir, f"{split}_fused_ids.npy")

        # 1. Try loading from cache
        if load_cached_data:
            if os.path.exists(path_X) and os.path.exists(path_ids):
                # Check y existence (not needed for test)
                if split == "test" or os.path.exists(path_y):
                    print(f"Loading processed fused data for '{split}' from cache...")
                    X = np.load(path_X)
                    ids = np.load(path_ids)
                    y = np.load(path_y) if split != "test" else None
                    return X, y, ids

        # 2. Process from scratch
        print(f"Processing data for split '{split}'...")

        # Ensure transformers are fitted (requires training data)
        self._ensure_fitted(load_cached_data=load_cached_data)

        # Load raw features for the requested split
        dino_feats, conv_feats, ids, labels = self.feature_extractor.extract_features(
            split=split, load_cached_data=load_cached_data
        )

        tab_feats, _, tab_ids = load_tabular_data(
            split=split, load_cached_data=load_cached_data
        )

        # Verify ID alignment between image and tabular streams
        if not np.array_equal(ids, tab_ids):
            raise ValueError(
                f"ID mismatch between image and tabular data for split {split}"
            )

        # Transform features
        print(f"  Transforming DINO features...")
        dino_trans = self.pca_dino.transform(dino_feats)

        print(f"  Transforming ConvNeXt features...")
        conv_trans = self.pca_conv.transform(conv_feats)

        print(f"  Transforming Tabular features...")
        tab_trans = self.qt_tabular.transform(tab_feats)

        # Fuse features
        print(f"  Fusing feature streams...")
        X_fused = np.concatenate([dino_trans, conv_trans, tab_trans], axis=1)

        print(f"  Final feature shape for {split}: {X_fused.shape}")

        # Save to cache
        print(f"Saving processed data to {cache_dir}...")
        np.save(path_X, X_fused)
        np.save(path_ids, ids)
        if labels is not None:
            np.save(path_y, labels)

        return X_fused, labels, ids
