import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from library.config import Config


class ManifoldProcessor:
    """
    Handles data topology transformations and feature engineering for the
    'View-Expanded Manifold Stabilization' strategy.

    Pipeline:
    1. Topology: Expand (Train) or Centroid (Val/Test)
    2. Tabular: QuantileTransform
    3. Fusion: Concatenate [Visual, Tabular]
    4. Projection: PCA
    """

    def __init__(self):
        # Initialize transformers
        # QuantileTransformer for tabular features to Gaussianize distributions
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        # PCA for dimensionality reduction on the fused feature vector
        # Retain 99% variance as per solution design
        self.pca = PCA(
            n_components=Config.PCA_VARIANCE,
            svd_solver="full",
            random_state=Config.SEED,
        )

        self.is_fitted = False

    def _get_cache_path(self, prefix, name):
        """Helper to construct cache file paths."""
        return os.path.join(Config.WORKING_DIR, f"{prefix}_{name}.npy")

    def create_expanded_set(self, data_dict, prefix="train", load_cache=True):
        """
        Creates the Expanded Training Set (4x size).
        Flattens (N, 4, D) -> (N*4, D) and replicates tabular/labels.
        """
        # Define cache paths
        path_emb = self._get_cache_path(prefix, "expanded_embeddings")
        path_tab = self._get_cache_path(prefix, "expanded_tabular")
        path_lbl = self._get_cache_path(prefix, "expanded_labels")
        path_ids = self._get_cache_path(prefix, "expanded_ids")

        # Try loading from cache
        if load_cache:
            if (
                os.path.exists(path_emb)
                and os.path.exists(path_tab)
                and os.path.exists(path_ids)
            ):
                # Check labels existence (test set might not have them)
                has_labels = os.path.exists(path_lbl)
                if not ("labels" in data_dict) or has_labels:
                    # print(f"Loading expanded set for '{prefix}' from cache...")
                    result = {
                        "embeddings": np.load(path_emb),
                        "tabular": np.load(path_tab),
                        "ids": np.load(path_ids),
                    }
                    if has_labels:
                        result["labels"] = np.load(path_lbl, allow_pickle=True)
                    return result

        # Compute from scratch
        # print(f"Creating expanded set for '{prefix}'...")
        embeddings = data_dict["embeddings"]  # (N, 4, D)
        tabular = data_dict["tabular"]  # (N, F)
        ids = data_dict["ids"]  # (N,)

        N, V, D = embeddings.shape

        # Flatten embeddings: (N, 4, D) -> (N*4, D)
        exp_embeddings = embeddings.reshape(N * V, D)

        # Replicate tabular: (N, F) -> (N*4, F)
        exp_tabular = np.repeat(tabular, V, axis=0)

        # Replicate ids: (N,) -> (N*4,)
        exp_ids = np.repeat(ids, V, axis=0)

        result = {"embeddings": exp_embeddings, "tabular": exp_tabular, "ids": exp_ids}

        # Replicate labels if present
        if "labels" in data_dict:
            labels = data_dict["labels"]
            exp_labels = np.repeat(labels, V, axis=0)
            result["labels"] = exp_labels
            np.save(path_lbl, exp_labels)

        # Save to cache
        np.save(path_emb, exp_embeddings)
        np.save(path_tab, exp_tabular)
        np.save(path_ids, exp_ids)

        return result

    def create_centroid_set(self, data_dict, prefix="val", load_cache=True):
        """
        Creates the Centroid Set (1x size).
        Computes mean of views (N, 4, D) -> (N, D).
        """
        # Define cache paths
        path_emb = self._get_cache_path(prefix, "centroid_embeddings")
        path_tab = self._get_cache_path(prefix, "centroid_tabular")
        path_lbl = self._get_cache_path(prefix, "centroid_labels")
        path_ids = self._get_cache_path(prefix, "centroid_ids")

        # Try loading from cache
        if load_cache:
            if (
                os.path.exists(path_emb)
                and os.path.exists(path_tab)
                and os.path.exists(path_ids)
            ):
                has_labels = os.path.exists(path_lbl)
                if not ("labels" in data_dict) or has_labels:
                    # print(f"Loading centroid set for '{prefix}' from cache...")
                    result = {
                        "embeddings": np.load(path_emb),
                        "tabular": np.load(path_tab),
                        "ids": np.load(path_ids),
                    }
                    if has_labels:
                        result["labels"] = np.load(path_lbl, allow_pickle=True)
                    return result

        # Compute from scratch
        # print(f"Creating centroid set for '{prefix}'...")
        embeddings = data_dict["embeddings"]  # (N, 4, D)
        tabular = data_dict["tabular"]  # (N, F)
        ids = data_dict["ids"]  # (N,)

        # Compute Mean: (N, 4, D) -> (N, D)
        cent_embeddings = np.mean(embeddings, axis=1)

        # Tabular and IDs remain unchanged (N, F) and (N,)
        cent_tabular = tabular
        cent_ids = ids

        result = {
            "embeddings": cent_embeddings,
            "tabular": cent_tabular,
            "ids": cent_ids,
        }

        if "labels" in data_dict:
            result["labels"] = data_dict["labels"]
            np.save(path_lbl, result["labels"])

        # Save to cache
        np.save(path_emb, cent_embeddings)
        np.save(path_tab, cent_tabular)
        np.save(path_ids, cent_ids)

        return result

    def fit_transform_train(self, raw_train_data, load_cache=True):
        """
        Full pipeline for Training Data:
        1. Expand Data
        2. Fit & Transform QuantileTransformer (Tabular)
        3. Fuse
        4. Fit & Transform PCA
        """
        # 1. Expand Data
        expanded_data = self.create_expanded_set(
            raw_train_data, prefix="train", load_cache=load_cache
        )

        X_img = expanded_data["embeddings"]
        X_tab = expanded_data["tabular"]
        y = expanded_data.get("labels", None)
        ids = expanded_data["ids"]

        # 2. Fit & Transform QuantileTransformer
        # print("Fitting QuantileTransformer on tabular data...")
        X_tab_trans = self.qt.fit_transform(X_tab)

        # 3. Fuse (Concatenate)
        # print("Fusing features...")
        X_fused = np.concatenate([X_img, X_tab_trans], axis=1)

        # 4. Fit & Transform PCA
        # print(f"Fitting PCA on fused data (Shape: {X_fused.shape})...")
        X_final = self.pca.fit_transform(X_fused)

        self.is_fitted = True
        # print(f"Training processing complete. Final Shape: {X_final.shape}")

        return X_final, y, ids

    def transform_inference(self, raw_data, prefix="val", load_cache=True):
        """
        Full pipeline for Inference (Val/Test):
        1. Centroid Data
        2. Transform QuantileTransformer (Tabular)
        3. Fuse
        4. Transform PCA
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ManifoldProcessor must be fitted on training data before inference."
            )

        # 1. Centroid Data
        centroid_data = self.create_centroid_set(
            raw_data, prefix=prefix, load_cache=load_cache
        )

        X_img = centroid_data["embeddings"]
        X_tab = centroid_data["tabular"]
        y = centroid_data.get("labels", None)
        ids = centroid_data["ids"]

        # 2. Transform QuantileTransformer
        X_tab_trans = self.qt.transform(X_tab)

        # 3. Fuse
        X_fused = np.concatenate([X_img, X_tab_trans], axis=1)

        # 4. Transform PCA
        X_final = self.pca.transform(X_fused)

        # print(f"Inference processing complete for '{prefix}'. Final Shape: {X_final.shape}")

        return X_final, y, ids
