import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything


class DataProcessor:
    """
    Handles data preparation for the Stratified Selective-Topology pipeline.
    Implements Orthogonal Manifold Densification by generating 3 orthogonal centroids
    per image and replicating tabular features to match.
    """

    def __init__(self):
        seed_everything(Config.SEED)

        # Feature dimensions based on the specified models in Config
        # DINOv2 ViT-Large: 1024
        # ConvNeXt Large: 1536
        # Tabular: 192 (64 margin + 64 shape + 64 texture)
        self.dino_dim = 1024
        self.conv_dim = 1536
        self.tab_dim = 192

    def get_column_indices(self):
        """
        Returns the column indices for each feature modality in the concatenated feature vector.
        Used by the model pipeline to apply selective transformations (PCA vs Quantile).

        Returns:
            dict: Mapping of feature type to list of indices.
        """
        dino_start = 0
        dino_end = self.dino_dim

        conv_start = dino_end
        conv_end = conv_start + self.conv_dim

        tab_start = conv_end
        tab_end = tab_start + self.tab_dim

        return {
            "dino": list(range(dino_start, dino_end)),
            "conv": list(range(conv_start, conv_end)),
            "tabular": list(range(tab_start, tab_end)),
        }

    def generate_orthogonal_centroids(self, features):
        """
        Averages raw 12-view features into 3 orthogonal centroids.

        Centroid A: Views {0, 3, 6, 9}   (0°, 90°, 180°, 270°)
        Centroid B: Views {1, 4, 7, 10}  (30°, 120°, 210°, 300°)
        Centroid C: Views {2, 5, 8, 11}  (60°, 150°, 240°, 330°)

        Args:
            features (np.ndarray): Raw features of shape [N, 12, D].

        Returns:
            np.ndarray: Centroids of shape [N, 3, D].
        """
        # Indices corresponding to the orthogonal views
        indices_a = [0, 3, 6, 9]
        indices_b = [1, 4, 7, 10]
        indices_c = [2, 5, 8, 11]

        # Average along the view dimension (axis 1)
        centroid_a = np.mean(features[:, indices_a, :], axis=1)  # [N, D]
        centroid_b = np.mean(features[:, indices_b, :], axis=1)  # [N, D]
        centroid_c = np.mean(features[:, indices_c, :], axis=1)  # [N, D]

        # Stack into [N, 3, D]
        centroids = np.stack([centroid_a, centroid_b, centroid_c], axis=1)
        return centroids

    def prepare_densified_dataset(
        self,
        dataset_name,
        dino_features=None,
        conv_features=None,
        ids=None,
        load_cached_data=True,
    ):
        """
        Prepares the densified dataset (3x size) by combining visual centroids and replicated tabular data.

        Logic:
            1. Load metadata to get tabular features and labels.
            2. Generate 3 centroids for visual features.
            3. Replicate tabular features and labels 3 times.
            4. Concatenate into a single matrix X.

        Args:
            dataset_name (str): 'train', 'val', or 'test'.
            dino_features (np.ndarray): Raw DINO features [N, 12, 1024].
            conv_features (np.ndarray): Raw ConvNeXt features [N, 12, 1536].
            ids (np.ndarray): Image IDs corresponding to the features [N].
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X, y, ids)
                X (np.ndarray): [3N, Total_Features]
                y (np.ndarray or None): [3N] labels
                ids (np.ndarray): [3N] replicated IDs
        """
        # Ensure cache directory exists
        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # Define cache file paths
        cache_X_path = os.path.join(cache_dir, f"densified_{dataset_name}_X.npy")
        cache_y_path = os.path.join(cache_dir, f"densified_{dataset_name}_y.npy")
        cache_ids_path = os.path.join(cache_dir, f"densified_{dataset_name}_ids.npy")

        # 1. Attempt to load from cache
        if load_cached_data:
            # Check if X and IDs exist. y is optional (for test set).
            if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
                # For train/val, y must also exist
                if dataset_name == "test" or os.path.exists(cache_y_path):
                    print(f"Loading densified {dataset_name} data from cache...")
                    X = np.load(cache_X_path)
                    ids_densified = np.load(cache_ids_path)

                    y = None
                    if os.path.exists(cache_y_path):
                        y = np.load(cache_y_path, allow_pickle=True)

                    return X, y, ids_densified

        # 2. Process from scratch
        print(f"Generating densified {dataset_name} data from scratch...")

        if dino_features is None or conv_features is None or ids is None:
            raise ValueError(
                f"Features and IDs must be provided to generate {dataset_name} data."
            )

        # Determine metadata path
        if dataset_name == "train":
            meta_path = Config.TRAIN_METADATA
        elif dataset_name == "val":
            meta_path = Config.VAL_METADATA
        elif dataset_name == "test":
            meta_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        # Load metadata
        df = pd.read_csv(meta_path)

        # Filter and align metadata to the provided IDs
        # This handles cases where 'limit' was used in feature extraction
        df = df[df["id"].isin(ids)]

        # Reorder dataframe to match the order of 'ids' array exactly
        df = df.set_index("id").reindex(ids).reset_index()

        # Extract Tabular Features
        # Columns: margin1..64, shape1..64, texture1..64
        margin_cols = [f"margin{i}" for i in range(1, 65)]
        shape_cols = [f"shape{i}" for i in range(1, 65)]
        texture_cols = [f"texture{i}" for i in range(1, 65)]
        tabular_cols = margin_cols + shape_cols + texture_cols

        # Shape: [N, 192]
        tabular_features = df[tabular_cols].values.astype(np.float32)

        # Extract Labels if present
        labels = None
        if "species" in df.columns:
            labels = df["species"].values  # Shape: [N]

        # Generate Orthogonal Centroids for Visual Features
        # Input: [N, 12, D] -> Output: [N, 3, D]
        dino_centroids = self.generate_orthogonal_centroids(dino_features)
        conv_centroids = self.generate_orthogonal_centroids(conv_features)

        # Densification: Flatten and Replicate
        # We flatten [N, 3, D] to [3N, D] such that order is:
        # Sample1_C1, Sample1_C2, Sample1_C3, Sample2_C1...
        dino_flat = dino_centroids.reshape(-1, self.dino_dim)
        conv_flat = conv_centroids.reshape(-1, self.conv_dim)

        # Replicate Tabular Features and IDs to match the visual centroids
        # np.repeat with axis=0 repeats rows: R1, R1, R1, R2, R2, R2...
        tab_flat = np.repeat(tabular_features, 3, axis=0)
        ids_flat = np.repeat(ids, 3, axis=0)

        if labels is not None:
            labels_flat = np.repeat(labels, 3, axis=0)
        else:
            labels_flat = None

        # Concatenate: [DINO | ConvNeXt | Tabular]
        # Shape: [3N, 1024 + 1536 + 192] = [3N, 2752]
        X = np.hstack([dino_flat, conv_flat, tab_flat])

        # Save to Cache
        print(f"Saving densified {dataset_name} data to {cache_dir}...")
        np.save(cache_X_path, X)
        np.save(cache_ids_path, ids_flat)
        if labels_flat is not None:
            np.save(cache_y_path, labels_flat)

        return X, labels_flat, ids_flat
