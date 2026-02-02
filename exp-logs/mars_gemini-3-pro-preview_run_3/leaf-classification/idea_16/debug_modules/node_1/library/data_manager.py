import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.feature_extractor import ImageEmbedder


class DensifiedDataLoader:
    """
    Manages data loading and topological transformations (Hyper-Densification and Canonical Centroids).
    """

    def __init__(self):
        self.embedder = ImageEmbedder()

    def _get_feature_columns(self):
        """Generates the list of 192 tabular feature column names."""
        margin_cols = [f"margin_{i}" for i in range(1, 65)]
        shape_cols = [f"shape_{i}" for i in range(1, 65)]
        texture_cols = [f"texture_{i}" for i in range(1, 65)]
        return margin_cols + shape_cols + texture_cols

    def load_tabular_data(self, csv_path):
        """
        Loads tabular features, labels, and IDs from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            tuple: (X_tab, y, ids)
                - X_tab (np.ndarray): (N, 192) float32 array of features.
                - y (np.ndarray or None): (N,) string array of species labels, or None if not present.
                - ids (np.ndarray): (N,) int array of image IDs.
        """
        df = pd.read_csv(csv_path)

        feature_cols = self._get_feature_columns()

        # Ensure all feature columns exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing feature columns in {csv_path}: {missing_cols[:5]}..."
            )

        X_tab = df[feature_cols].values.astype(np.float32)
        ids = df["id"].values.astype(np.int32)

        if "species" in df.columns:
            y = df["species"].values.astype(str)
        else:
            y = None

        return X_tab, y, ids

    def generate_train_data(self, load_cached_data=True):
        """
        Generates the Hyper-Densified training dataset.
        Creates 9 orthogonal centroids per image by averaging mutually exclusive sets of 4 views.
        Replicates tabular data and labels to match.

        Args:
            load_cached_data (bool): Whether to load from disk if available.

        Returns:
            dict: Dictionary containing:
                - 'dino': (N*9, D_dino)
                - 'convnext': (N*9, D_conv)
                - 'tabular': (N*9, 192)
                - 'y': (N*9,) labels
                - 'ids': (N*9,) ids
        """
        seed_everything()

        # Cache paths
        cache_paths = {
            "dino": os.path.join(Config.CACHE_DIR, "train_densified_dino.npy"),
            "convnext": os.path.join(Config.CACHE_DIR, "train_densified_convnext.npy"),
            "tabular": os.path.join(Config.CACHE_DIR, "train_densified_tabular.npy"),
            "y": os.path.join(Config.CACHE_DIR, "train_densified_y.npy"),
            "ids": os.path.join(Config.CACHE_DIR, "train_densified_ids.npy"),
        }

        # 1. Try Load from Cache
        if load_cached_data:
            if all(os.path.exists(p) for p in cache_paths.values()):
                print("[Train] Loading densified data from cache...")
                return {
                    k: np.load(p, allow_pickle=True) for k, p in cache_paths.items()
                }
            else:
                print("[Train] Cache missing or incomplete. Regenerating data...")

        # 2. Generate Data
        # Get raw multi-view features: (N, 36, D)
        raw_features = self.embedder.extract_features(
            dataset_name="train",
            csv_path=Config.TRAIN_CSV,
            load_cached_data=load_cached_data,
        )

        dino_raw = raw_features["dino"]
        conv_raw = raw_features["convnext"]

        # Get tabular data
        X_tab, y, ids = self.load_tabular_data(Config.TRAIN_CSV)

        # Verify alignment
        if len(ids) != len(dino_raw):
            raise ValueError("Mismatch between CSV rows and extracted features.")

        # --- Densification Logic ---
        # We want 9 centroids. Each centroid k averages views [k, k+9, k+18, k+27].
        # Current shape: (N, 36, D).
        # We reshape to (N, 4, 9, D).
        #   Dim 1 (size 4) represents the 4 orthogonal views for a specific centroid.
        #   Dim 2 (size 9) represents the 9 distinct centroids.
        #   Example: [n, 0, k, :] -> view k
        #            [n, 1, k, :] -> view k+9
        #            [n, 2, k, :] -> view k+18
        #            [n, 3, k, :] -> view k+27

        N, V, D_dino = dino_raw.shape
        _, _, D_conv = conv_raw.shape

        n_centroids = Config.N_CENTROIDS_TRAIN  # 9
        views_per_centroid = Config.VIEWS_PER_CENTROID  # 4

        # Reshape and Mean
        # (N, 36, D) -> (N, 4, 9, D) -> Mean(axis=1) -> (N, 9, D)
        dino_densified = dino_raw.reshape(
            N, views_per_centroid, n_centroids, D_dino
        ).mean(axis=1)
        conv_densified = conv_raw.reshape(
            N, views_per_centroid, n_centroids, D_conv
        ).mean(axis=1)

        # Flatten to (N*9, D)
        dino_final = dino_densified.reshape(N * n_centroids, D_dino)
        conv_final = conv_densified.reshape(N * n_centroids, D_conv)

        # Replicate Tabular, Y, IDs
        # (N, F) -> (N, 9, F) -> (N*9, F)
        tab_final = np.repeat(X_tab[:, np.newaxis, :], n_centroids, axis=1).reshape(
            N * n_centroids, -1
        )
        y_final = np.repeat(y[:, np.newaxis], n_centroids, axis=1).reshape(
            N * n_centroids
        )
        ids_final = np.repeat(ids[:, np.newaxis], n_centroids, axis=1).reshape(
            N * n_centroids
        )

        data = {
            "dino": dino_final,
            "convnext": conv_final,
            "tabular": tab_final,
            "y": y_final,
            "ids": ids_final,
        }

        # 3. Save to Cache
        print("[Train] Saving densified data to cache...")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        for k, v in data.items():
            np.save(cache_paths[k], v)

        return data

    def generate_inference_data(self, dataset_name, csv_path, load_cached_data=True):
        """
        Generates the Canonical Centroid dataset for inference (Validation or Test).
        Creates 1 centroid per image by averaging the 4 canonical views (0, 90, 180, 270 degrees).

        Args:
            dataset_name (str): Name of the dataset (e.g., 'val', 'test').
            csv_path (str): Path to the CSV file.
            load_cached_data (bool): Whether to load from disk if available.

        Returns:
            dict: Dictionary containing:
                - 'dino': (N, D_dino)
                - 'convnext': (N, D_conv)
                - 'tabular': (N, 192)
                - 'y': (N,) labels (or None for test)
                - 'ids': (N,) ids
        """
        seed_everything()

        # Cache paths
        cache_paths = {
            "dino": os.path.join(
                Config.CACHE_DIR, f"{dataset_name}_canonical_dino.npy"
            ),
            "convnext": os.path.join(
                Config.CACHE_DIR, f"{dataset_name}_canonical_convnext.npy"
            ),
            "tabular": os.path.join(
                Config.CACHE_DIR, f"{dataset_name}_canonical_tabular.npy"
            ),
            "y": os.path.join(Config.CACHE_DIR, f"{dataset_name}_canonical_y.npy"),
            "ids": os.path.join(Config.CACHE_DIR, f"{dataset_name}_canonical_ids.npy"),
        }

        # 1. Try Load from Cache
        if load_cached_data:
            # Check existence (y might not exist for test, so we handle it conditionally)
            required_keys = ["dino", "convnext", "tabular", "ids"]
            if dataset_name != "test":
                required_keys.append("y")

            if all(os.path.exists(cache_paths[k]) for k in required_keys):
                print(f"[{dataset_name}] Loading canonical data from cache...")
                loaded_data = {
                    k: np.load(cache_paths[k], allow_pickle=True) for k in required_keys
                }
                # For test, y is None
                if dataset_name == "test":
                    loaded_data["y"] = None
                return loaded_data
            else:
                print(
                    f"[{dataset_name}] Cache missing or incomplete. Regenerating data..."
                )

        # 2. Generate Data
        # Get raw multi-view features: (N, 36, D)
        raw_features = self.embedder.extract_features(
            dataset_name=dataset_name,
            csv_path=csv_path,
            load_cached_data=load_cached_data,
        )

        dino_raw = raw_features["dino"]
        conv_raw = raw_features["convnext"]

        # Get tabular data
        X_tab, y, ids = self.load_tabular_data(csv_path)

        # Verify alignment
        if len(ids) != len(dino_raw):
            raise ValueError("Mismatch between CSV rows and extracted features.")

        # --- Canonical Transformation Logic ---
        # Indices: [0, 9, 18, 27]
        indices = Config.CANONICAL_INDICES

        # Select views and Mean: (N, 36, D) -> (N, 4, D) -> (N, D)
        dino_canonical = dino_raw[:, indices, :].mean(axis=1)
        conv_canonical = conv_raw[:, indices, :].mean(axis=1)

        data = {
            "dino": dino_canonical,
            "convnext": conv_canonical,
            "tabular": X_tab,
            "y": y,
            "ids": ids,
        }

        # 3. Save to Cache
        print(f"[{dataset_name}] Saving canonical data to cache...")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        np.save(cache_paths["dino"], data["dino"])
        np.save(cache_paths["convnext"], data["convnext"])
        np.save(cache_paths["tabular"], data["tabular"])
        np.save(cache_paths["ids"], data["ids"])

        if data["y"] is not None:
            np.save(cache_paths["y"], data["y"])

        return data
