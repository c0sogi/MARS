import os
import numpy as np
import pandas as pd
from library.config import Config


class ManifoldDensifier:
    """
    Handles the geometric manipulation of data manifolds.
    Aggregates multi-view features into orthogonal centroids and densifies
    the dataset for training by treating centroids as distinct samples.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.tabular_prefixes = Config.TABULAR_COLS_PREFIXES
        # Explicit order to ensure alignment with np.repeat and reshape operations
        self.centroid_keys = ["A", "B", "C"]

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def compute_centroids(self, features):
        """
        Aggregates 12 views into 3 orthogonal centroids based on Config.

        Args:
            features (np.ndarray): Input features of shape (N, 12, D).

        Returns:
            np.ndarray: Centroids of shape (N, 3, D).
        """
        N, V, D = features.shape
        centroid_list = []

        for key in self.centroid_keys:
            indices = Config.CENTROID_INDICES[key]
            # Select specific views: (N, 4, D)
            view_subset = features[:, indices, :]
            # Compute mean: (N, D)
            centroid = np.mean(view_subset, axis=1)
            centroid_list.append(centroid)

        # Stack along axis 1: (N, 3, D)
        # Order corresponds to A, B, C
        return np.stack(centroid_list, axis=1)

    def _get_tabular_features(self, df):
        """
        Extracts the 192 tabular features (margin, shape, texture) from the dataframe.
        """
        cols = []
        # We assume the columns in the CSV are already in a consistent order.
        # We collect all columns starting with the defined prefixes.
        for prefix in self.tabular_prefixes:
            # Filter columns starting with prefix (e.g., 'margin')
            # Note: This relies on the column naming convention in the dataset
            current_cols = [c for c in df.columns if c.startswith(prefix)]
            cols.extend(current_cols)

        return df[cols].values.astype(np.float32)

    def process_split(
        self, split_name, dino_feats, conv_feats, ids, load_cached_data=True
    ):
        """
        Main processing pipeline for a dataset split.
        1. Checks cache.
        2. Loads metadata and aligns with provided IDs.
        3. Computes centroids.
        4. Densifies (flattens) the data structure.
        5. Caches and returns.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            dino_feats (np.ndarray): (N, 12, D1) DINOv2 features.
            conv_feats (np.ndarray): (N, 12, D2) ConvNeXt features.
            ids (np.ndarray): (N,) Image IDs corresponding to features.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            Tuple: (dino_dense, conv_dense, tab_dense, ids_dense, y_dense)
            Shapes will be (3*N, ...)
        """
        # Define cache paths
        cache_prefix = os.path.join(self.working_dir, f"{split_name}_densified")
        path_dino = f"{cache_prefix}_dino.npy"
        path_conv = f"{cache_prefix}_conv.npy"
        path_tab = f"{cache_prefix}_tab.npy"
        path_ids = f"{cache_prefix}_ids.npy"
        path_y = f"{cache_prefix}_y.npy"

        # 1. Try Loading Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_dino)
                and os.path.exists(path_conv)
                and os.path.exists(path_tab)
                and os.path.exists(path_ids)
            )
            # For test set, y might not exist, so we check y only if we expect it?
            # Simpler: check if the main feature files exist.

            if files_exist:
                print(f"[{split_name}] Loading densified manifold data from cache...")
                dino_dense = np.load(path_dino)
                conv_dense = np.load(path_conv)
                tab_dense = np.load(path_tab)
                ids_dense = np.load(path_ids)

                y_dense = None
                if os.path.exists(path_y):
                    y_dense = np.load(path_y, allow_pickle=True)

                return dino_dense, conv_dense, tab_dense, ids_dense, y_dense

        # 2. Process from Scratch
        print(f"[{split_name}] Computing orthogonal centroids and densifying data...")

        # Determine metadata path
        if split_name == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split_name == "val":
            meta_path = Config.VAL_METADATA_PATH
        else:
            meta_path = Config.TEST_METADATA_PATH

        # Load metadata
        df_meta = pd.read_csv(meta_path)

        # Align metadata to the order of 'ids' provided by FeatureExtractor
        # The FeatureExtractor might return a subset (debug) or specific order.
        # We reindex the dataframe to match 'ids'.
        df_meta = df_meta.set_index("id")
        try:
            df_aligned = df_meta.loc[ids].reset_index()
        except KeyError as e:
            raise KeyError(
                f"Feature IDs not found in metadata {meta_path}. Ensure metadata and features are consistent."
            ) from e

        # Extract Tabular Features
        tab_features = self._get_tabular_features(df_aligned)  # (N, 192)

        # Extract Labels if available
        labels = None
        if "species" in df_aligned.columns:
            labels = df_aligned["species"].values  # (N,)

        # Compute Centroids (N, 12, D) -> (N, 3, D)
        dino_centroids = self.compute_centroids(dino_feats)
        conv_centroids = self.compute_centroids(conv_feats)

        # Densify / Flatten
        # We flatten the 3 centroids into distinct samples.
        # (N, 3, D) -> (3*N, D)
        # The reshape order (default C-contiguous) will iterate the last axis, then the second to last.
        # So it goes: ID0_CentroidA, ID0_CentroidB, ID0_CentroidC, ID1_CentroidA...
        N, _, D_dino = dino_centroids.shape
        _, _, D_conv = conv_centroids.shape

        dino_dense = dino_centroids.reshape(-1, D_dino)
        conv_dense = conv_centroids.reshape(-1, D_conv)

        # Replicate Tabular, IDs, and Labels to match the flattened structure
        # np.repeat with repeats=3 produces: ID0, ID0, ID0, ID1, ID1, ID1...
        # This matches the reshape order of the centroids.
        tab_dense = np.repeat(tab_features, 3, axis=0)
        ids_dense = np.repeat(ids, 3, axis=0)

        y_dense = None
        if labels is not None:
            y_dense = np.repeat(labels, 3, axis=0)

        # 3. Save to Cache
        print(f"[{split_name}] Saving densified artifacts to {self.working_dir}")
        np.save(path_dino, dino_dense)
        np.save(path_conv, conv_dense)
        np.save(path_tab, tab_dense)
        np.save(path_ids, ids_dense)
        if y_dense is not None:
            np.save(path_y, y_dense)

        return dino_dense, conv_dense, tab_dense, ids_dense, y_dense
