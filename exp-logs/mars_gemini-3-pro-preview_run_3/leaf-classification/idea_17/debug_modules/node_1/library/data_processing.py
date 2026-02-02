import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from library.config import Config
from library.feature_extraction import FeatureExtractor
from library.utils import seed_everything


class LeafDataProcessor:
    """
    Manages data loading, splitting, hyper-densification, and preprocessing
    for the Stacked Discriminant Analysis pipeline.
    """

    def __init__(self, load_raw_cache=True):
        """
        Initializes the processor by loading raw features from the FeatureExtractor.
        Merges the provided 'train' and 'val' splits to allow for custom Stratified K-Fold.
        """
        seed_everything(Config.SEED)

        # Load raw features (using FeatureExtractor's caching mechanism)
        extractor = FeatureExtractor()
        raw_data = extractor.extract_all(load_cached_data=load_raw_cache)

        # Merge original Train and Val for Stratified K-Fold
        self.X_dino_full = np.concatenate(
            [raw_data["train_dino"], raw_data["val_dino"]], axis=0
        )
        self.X_conv_full = np.concatenate(
            [raw_data["train_conv"], raw_data["val_conv"]], axis=0
        )
        self.X_tab_full = np.concatenate(
            [raw_data["train_tab"], raw_data["val_tab"]], axis=0
        )
        self.ids_full = np.concatenate(
            [raw_data["train_ids"], raw_data["val_ids"]], axis=0
        )
        self.y_full = np.concatenate(
            [raw_data["train_lbl"], raw_data["val_lbl"]], axis=0
        )

        # Test data (always kept separate)
        self.X_dino_test = raw_data["test_dino"]
        self.X_conv_test = raw_data["test_conv"]
        self.X_tab_test = raw_data["test_tab"]
        self.ids_test = raw_data["test_ids"]

        # Encode labels globally to ensure consistency across folds
        self.label_encoder = LabelEncoder()
        self.y_encoded_full = self.label_encoder.fit_transform(self.y_full)
        self.classes = self.label_encoder.classes_

        # Initialize K-Fold splitter
        self.skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )

    def _densify_images(self, features, mode):
        """
        Applies Hyper-Densification to image features.

        Args:
            features (np.ndarray): Shape (N, 36, D)
            mode (str): 'train' (9 centroids) or 'inference' (1 canonical centroid)

        Returns:
            np.ndarray: Densified features.
                        Shape (N*9, D) for 'train'.
                        Shape (N, D) for 'inference'.
        """
        N, V, D = features.shape

        if mode == "train":
            densified = []
            # Generate 9 orthogonal centroids
            for k in range(Config.NUM_TRAIN_CENTROIDS):
                indices = Config.get_centroid_indices(k)
                # Average across the 4 views for this centroid
                centroid = np.mean(features[:, indices, :], axis=1)  # (N, D)
                densified.append(centroid)

            # Stack: We want to interleave or stack?
            # Stacking vertically: [Centroid_0_All_Imgs, Centroid_1_All_Imgs, ...]
            # This is easier for batch processing.
            return np.concatenate(densified, axis=0)  # (N*9, D)

        elif mode == "inference":
            indices = Config.get_canonical_indices()
            centroid = np.mean(features[:, indices, :], axis=1)  # (N, D)
            return centroid

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _densify_tabular(self, features, mode):
        """
        Replicates tabular features for training densification.
        """
        if mode == "train":
            # Replicate features 9 times to match image centroids
            return np.tile(features, (Config.NUM_TRAIN_CENTROIDS, 1))
        return features

    def _densify_meta(self, arr, mode):
        """
        Replicates metadata (ids, labels) for training densification.
        """
        if mode == "train":
            return np.tile(arr, Config.NUM_TRAIN_CENTROIDS)
        return arr

    def get_fold_data(self, fold_idx, load_cache=True):
        """
        Retrieves processed data for a specific fold.
        Handles caching, splitting, densification, PCA, and Quantile transformation.

        Args:
            fold_idx (int): Index of the fold (0 to NUM_FOLDS-1).
            load_cache (bool): Whether to attempt loading from cache.

        Returns:
            dict: Contains processed training, validation, and test data for the fold.
        """
        Config.setup_directories()
        cache_file = os.path.join(Config.CACHE_DIR, f"fold_{fold_idx}_data.npy")

        # 1. Try Load Cache
        if load_cache and os.path.exists(cache_file):
            print(f"Loading processed data for Fold {fold_idx} from cache...")
            return np.load(cache_file, allow_pickle=True).item()

        print(f"Processing Fold {fold_idx}...")

        # 2. Split Data
        # Get indices for this fold
        splits = list(self.skf.split(self.X_dino_full, self.y_encoded_full))
        train_idx, val_idx = splits[fold_idx]

        # Raw Fold Data
        X_dino_tr_raw = self.X_dino_full[train_idx]
        X_conv_tr_raw = self.X_conv_full[train_idx]
        X_tab_tr_raw = self.X_tab_full[train_idx]
        y_tr_raw = self.y_encoded_full[train_idx]
        ids_tr_raw = self.ids_full[train_idx]

        X_dino_val_raw = self.X_dino_full[val_idx]
        X_conv_val_raw = self.X_conv_full[val_idx]
        X_tab_val_raw = self.X_tab_full[val_idx]
        y_val_raw = self.y_encoded_full[val_idx]
        ids_val_raw = self.ids_full[val_idx]

        # 3. Hyper-Densification
        # Train: 9 centroids
        X_dino_tr = self._densify_images(X_dino_tr_raw, mode="train")
        X_conv_tr = self._densify_images(X_conv_tr_raw, mode="train")
        X_tab_tr = self._densify_tabular(X_tab_tr_raw, mode="train")
        y_tr = self._densify_meta(y_tr_raw, mode="train")
        ids_tr = self._densify_meta(ids_tr_raw, mode="train")

        # Val: 1 canonical centroid
        X_dino_val = self._densify_images(X_dino_val_raw, mode="inference")
        X_conv_val = self._densify_images(X_conv_val_raw, mode="inference")
        X_tab_val = self._densify_tabular(X_tab_val_raw, mode="inference")
        y_val = y_val_raw  # No replication
        ids_val = ids_val_raw

        # Test: 1 canonical centroid
        X_dino_test = self._densify_images(self.X_dino_test, mode="inference")
        X_conv_test = self._densify_images(self.X_conv_test, mode="inference")
        X_tab_test = self._densify_tabular(self.X_tab_test, mode="inference")
        ids_test = self.ids_test

        # 4. Preprocessing (Fit on Train, Transform All)

        # PCA for DINO
        pca_dino = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
        X_dino_tr = pca_dino.fit_transform(X_dino_tr)
        X_dino_val = pca_dino.transform(X_dino_val)
        X_dino_test = pca_dino.transform(X_dino_test)

        # PCA for ConvNeXt
        pca_conv = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)
        X_conv_tr = pca_conv.fit_transform(X_conv_tr)
        X_conv_val = pca_conv.transform(X_conv_val)
        X_conv_test = pca_conv.transform(X_conv_test)

        # QuantileTransformer for Tabular
        qt = QuantileTransformer(output_distribution="normal", random_state=Config.SEED)
        X_tab_tr = qt.fit_transform(X_tab_tr)
        X_tab_val = qt.transform(X_tab_val)
        X_tab_test = qt.transform(X_tab_test)

        # 5. Construct Result Dictionary
        fold_data = {
            "X_dino_train": X_dino_tr.astype(np.float32),
            "X_conv_train": X_conv_tr.astype(np.float32),
            "X_tab_train": X_tab_tr.astype(np.float32),
            "y_train": y_tr,
            "ids_train": ids_tr,
            "X_dino_val": X_dino_val.astype(np.float32),
            "X_conv_val": X_conv_val.astype(np.float32),
            "X_tab_val": X_tab_val.astype(np.float32),
            "y_val": y_val,
            "ids_val": ids_val,
            "X_dino_test": X_dino_test.astype(np.float32),
            "X_conv_test": X_conv_test.astype(np.float32),
            "X_tab_test": X_tab_test.astype(np.float32),
            "ids_test": ids_test,
            "classes": self.classes,
        }

        # 6. Save to Cache
        np.save(cache_file, fold_data)

        return fold_data

    def get_all_classes(self):
        """Returns the list of class names."""
        return self.classes
