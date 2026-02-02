import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_logger, load_data_from_cache, save_data_to_cache
from library.feature_extractor import FeatureExtractor

# Initialize logger
logger = get_logger(name="data_processor")


class LeafDataManager:
    """
    Manages data loading, processing, and manifold densification for the Leaf Classification task.

    Key Responsibilities:
    1. Load metadata for train, validation, and test sets.
    2. Invoke FeatureExtractor to get raw DINOv2 and ConvNeXt features (12 views).
    3. Implement Manifold Densification: Aggregate 12 views into 3 Orthogonal Centroids.
    4. Construct final feature matrices (X) by concatenating Visual and Tabular streams.
    5. Provide Stratified K-Fold splits for Cross-Validation.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.working_dir = Config.WORKING_DIR

        # Define feature dimensions
        self.dino_dim = 1024
        self.conv_dim = 1536
        self.tabular_dim = Config.TABULAR_FEATURE_COUNT

        # Define column slices for downstream pipelines
        # Order: [DINO (1024) | ConvNeXt (1536) | Tabular (192)]
        self.feature_slices = {
            "dino": slice(0, self.dino_dim),
            "conv": slice(self.dino_dim, self.dino_dim + self.conv_dim),
            "tabular": slice(
                self.dino_dim + self.conv_dim,
                self.dino_dim + self.conv_dim + self.tabular_dim,
            ),
        }

    def get_feature_slices(self):
        """Returns the slice indices for each feature modality in the concatenated matrix."""
        return self.feature_slices

    def _load_metadata(self, stage="train"):
        """
        Loads and merges metadata CSVs based on the stage.

        Args:
            stage (str): 'train' (merges train+val) or 'test'.

        Returns:
            pd.DataFrame: The loaded metadata.
        """
        if stage == "train":
            # Merge train and val for full K-Fold CV
            df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
            df_val = pd.read_csv(Config.VAL_METADATA_PATH)
            df = pd.concat([df_train, df_val], axis=0, ignore_index=True)
            logger.info(f"Loaded merged Train+Val metadata: {len(df)} samples")
        elif stage == "test":
            df = pd.read_csv(Config.TEST_METADATA_PATH)
            logger.info(f"Loaded Test metadata: {len(df)} samples")
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return df

    def _compute_centroids(self, features):
        """
        Aggregates 12-view features into 3 orthogonal centroids.

        Args:
            features (np.ndarray): Shape (N, 12, D)

        Returns:
            np.ndarray: Shape (N, 3, D)
        """
        N, V, D = features.shape

        # Extract views for each centroid based on indices in Config
        # Centroid A: Indices [0, 3, 6, 9]
        views_a = features[:, Config.CENTROID_A_INDICES, :]  # (N, 4, D)
        centroid_a = np.mean(views_a, axis=1)  # (N, D)

        # Centroid B: Indices [1, 4, 7, 10]
        views_b = features[:, Config.CENTROID_B_INDICES, :]
        centroid_b = np.mean(views_b, axis=1)

        # Centroid C: Indices [2, 5, 8, 11]
        views_c = features[:, Config.CENTROID_C_INDICES, :]
        centroid_c = np.mean(views_c, axis=1)

        # Stack centroids: (N, 3, D)
        centroids = np.stack([centroid_a, centroid_b, centroid_c], axis=1)
        return centroids

    def _densify_and_assemble(
        self, ids, dino_feats, conv_feats, tabular_df, labels=None
    ):
        """
        Performs Manifold Densification and assembles the final feature matrix.

        Logic:
        1. Compute 3 centroids for visual features.
        2. Flatten visual features: (N, 3, D) -> (N*3, D).
        3. Replicate tabular features 3 times to match.
        4. Concatenate [DINO, Conv, Tabular].

        Args:
            ids (np.ndarray): (N,)
            dino_feats (np.ndarray): (N, 12, 1024)
            conv_feats (np.ndarray): (N, 12, 1536)
            tabular_df (pd.DataFrame): (N, 192+) containing feature columns
            labels (np.ndarray, optional): (N,)

        Returns:
            tuple: (X, y, expanded_ids)
                X (np.ndarray): (N*3, Total_Dim)
                y (np.ndarray or None): (N*3,)
                expanded_ids (np.ndarray): (N*3,)
        """
        N = len(ids)

        # 1. Compute Centroids
        dino_centroids = self._compute_centroids(dino_feats)  # (N, 3, 1024)
        conv_centroids = self._compute_centroids(conv_feats)  # (N, 3, 1536)

        # 2. Flatten Centroids (Densification)
        # Reshape to (N*3, D)
        dino_flat = dino_centroids.reshape(N * 3, -1)
        conv_flat = conv_centroids.reshape(N * 3, -1)

        # 3. Process Tabular Data
        # Extract only the relevant feature columns
        # Columns: margin_1..64, shape_1..64, texture_1..64
        # We assume the dataframe passed here is aligned with 'ids'
        # To ensure alignment, we reindex or assumed caller handled it.
        # FeatureExtractor returns ids, we filtered tabular_df to match those ids in order.

        feat_cols = [
            c
            for c in tabular_df.columns
            if c.startswith(("margin", "shape", "texture"))
        ]
        if len(feat_cols) != self.tabular_dim:
            logger.warning(
                f"Expected {self.tabular_dim} tabular features, found {len(feat_cols)}"
            )

        tabular_vals = tabular_df[feat_cols].values  # (N, 192)

        # Replicate tabular features for each of the 3 centroids
        # np.repeat repeats elements, so we repeat rows 3 times
        # row 0 -> row 0, row 0, row 0
        tabular_flat = np.repeat(tabular_vals, 3, axis=0)  # (N*3, 192)

        # 4. Concatenate All Streams
        # Order: DINO, Conv, Tabular
        X = np.concatenate([dino_flat, conv_flat, tabular_flat], axis=1)

        # 5. Handle IDs and Labels
        expanded_ids = np.repeat(ids, 3)

        expanded_labels = None
        if labels is not None:
            expanded_labels = np.repeat(labels, 3)

        return X, expanded_labels, expanded_ids

    def get_dataset(self, stage="train", load_cached_data=True):
        """
        Main entry point to get processed, densified data.

        Args:
            stage (str): 'train' or 'test'.
            load_cached_data (bool): Whether to use disk caching.

        Returns:
            dict: {
                "X": np.ndarray (Features),
                "y": np.ndarray (Labels) or None,
                "ids": np.ndarray (Image IDs)
            }
        """
        # Define cache paths for the FINAL densified matrices
        cache_prefix = f"densified_{stage}"
        if Config.DEBUG:
            cache_prefix += "_debug"

        path_X = os.path.join(self.working_dir, f"{cache_prefix}_X.npy")
        path_y = os.path.join(self.working_dir, f"{cache_prefix}_y.npy")
        path_ids = os.path.join(self.working_dir, f"{cache_prefix}_ids.npy")

        # 1. Try Load from Cache
        if load_cached_data:
            X = load_data_from_cache(path_X)
            ids = load_data_from_cache(path_ids)
            y = load_data_from_cache(path_y) if stage == "train" else None

            # Check if loaded successfully
            if X is not None and ids is not None:
                if stage == "train" and y is not None:
                    logger.info(f"Loaded densified {stage} data from cache.")
                    return {"X": X, "y": y, "ids": ids}
                elif stage == "test":
                    logger.info(f"Loaded densified {stage} data from cache.")
                    return {"X": X, "y": None, "ids": ids}

        logger.info(f"Constructing {stage} dataset from scratch...")

        # 2. Load Metadata
        df = self._load_metadata(stage)

        # 3. Extract Raw Features (12 views)
        # This step uses its own caching mechanism within FeatureExtractor
        ids, dino_feats, conv_feats = self.feature_extractor.extract_features(
            df, dataset_name=stage, load_cached_data=load_cached_data
        )

        # 4. Filter Tabular Data to match Extracted IDs
        # FeatureExtractor might have dropped images if they failed to load (unlikely)
        # or if DEBUG mode sliced the dataframe differently.
        # We must align the tabular dataframe to the returned 'ids'.

        # Set ID as index for easy lookup
        df_indexed = df.set_index("id")

        # Reindex based on returned ids to ensure 1-to-1 correspondence and order
        df_aligned = df_indexed.loc[ids].reset_index()

        # Extract labels if training
        labels = None
        if stage == "train":
            labels = df_aligned["species"].values

        # 5. Densify and Assemble
        X, y_densified, ids_densified = self._densify_and_assemble(
            ids, dino_feats, conv_feats, df_aligned, labels
        )

        # 6. Save to Cache
        save_data_to_cache(X, path_X)
        save_data_to_cache(ids_densified, path_ids)
        if y_densified is not None:
            save_data_to_cache(y_densified, path_y)

        logger.info(f"Dataset {stage} construction complete. Shape: {X.shape}")

        return {"X": X, "y": y_densified, "ids": ids_densified}

    def get_stratified_folds(self, X, y, ids, n_folds=Config.N_FOLDS):
        """
        Generates stratified folds based on UNIQUE image IDs.

        Crucial: We must not split different centroids of the same image into different folds.
        We split based on the original image ID, then map those splits to the densified indices.

        Args:
            X (np.ndarray): Densified features (N*3, D)
            y (np.ndarray): Densified labels (N*3,)
            ids (np.ndarray): Densified IDs (N*3,)
            n_folds (int): Number of folds

        Yields:
            (train_indices, val_indices) for the densified arrays.
        """
        # 1. Identify unique IDs and their corresponding labels
        # Since ids and y are repeated 3 times, we can just take every 3rd element
        # assuming the structure [ID1_A, ID1_B, ID1_C, ID2_A, ...]

        # Verification of structure
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        # np.unique sorts the output, which might scramble alignment with y if not careful.
        # Instead, let's just slice.

        # We know the data was constructed by repeating 3 times.
        # Slice: start=0, step=3
        canonical_ids = ids[::3]
        canonical_y = y[::3]

        # 2. Perform Stratified Split on Canonical Data
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        for fold, (train_idx_canonical, val_idx_canonical) in enumerate(
            skf.split(canonical_ids, canonical_y)
        ):
            # 3. Map Canonical Indices back to Densified Indices
            # If index i is in train, then indices 3*i, 3*i+1, 3*i+2 are in train

            # Vectorized mapping
            train_indices = np.concatenate(
                [
                    train_idx_canonical * 3,
                    train_idx_canonical * 3 + 1,
                    train_idx_canonical * 3 + 2,
                ]
            )

            val_indices = np.concatenate(
                [
                    val_idx_canonical * 3,
                    val_idx_canonical * 3 + 1,
                    val_idx_canonical * 3 + 2,
                ]
            )

            # Sort to maintain order (optional but good for debugging)
            train_indices.sort()
            val_indices.sort()

            yield train_indices, val_indices
