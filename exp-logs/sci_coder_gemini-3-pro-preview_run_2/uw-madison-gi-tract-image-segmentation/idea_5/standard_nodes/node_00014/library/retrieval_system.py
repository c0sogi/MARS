import numpy as np
import pandas as pd
from library.config import Config
from library.data_processing import prepare_atlas_data, generate_search_vector


class AtlasSegmenter:
    """
    Retrieval-based Multi-Atlas Segmentation model.
    Uses a database of training images (Atlas) to segment new images by:
    1. Finding similar images in the training set (based on visual similarity and depth).
    2. Retrieving their ground truth masks.
    3. Fusing them via majority voting.
    """

    def __init__(self, config=Config):
        self.config = config
        self.k_neighbors = config.K_NEIGHBORS
        self.depth_tolerance = config.DEPTH_TOLERANCE
        self.fusion_threshold = config.FUSION_THRESHOLD

        # Atlas Index Storage
        self.atlas_vectors = None  # Shape: (N, Feature_Dim)
        self.atlas_masks = None  # Shape: (N, H, W, C)
        self.atlas_depths = None  # Shape: (N,)
        self.atlas_indices = None  # Shape: (N,) - Original indices mapping

    def fit(self, load_cached_data=True, debug=False):
        """
        Builds the search index by loading or processing the atlas data.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): Whether to run in debug mode (subset of data).
        """
        print("Initializing Atlas Search Index...")

        # Delegate data loading/processing to the library function
        # This function handles the caching logic (parquet/npy) internally
        index_df, vectors, masks = prepare_atlas_data(
            load_cached_data=load_cached_data, debug=debug
        )

        self.atlas_vectors = vectors
        self.atlas_masks = masks

        # Extract depths for fast filtering
        if "relative_depth" in index_df.columns:
            self.atlas_depths = index_df["relative_depth"].values.astype(np.float32)
        else:
            raise ValueError("Atlas metadata missing 'relative_depth' column.")

        # Keep track of original indices to map back to masks/vectors after filtering
        self.atlas_indices = np.arange(len(index_df))

        print(f"Atlas Index built with {len(self.atlas_indices)} slices.")

    def predict_slice(self, img, depth):
        """
        Predicts the segmentation mask for a single slice.

        Args:
            img (np.ndarray): Preprocessed input image (normalized).
            depth (float): Relative depth of the slice (0.0 to 1.0).

        Returns:
            np.ndarray: Predicted binary mask of shape (H, W, C).
        """
        if self.atlas_vectors is None:
            raise RuntimeError("AtlasSegmenter must be fit before prediction.")

        # 1. Generate Query Vector
        # Downsample image to create the search vector
        query_vec = generate_search_vector(img, self.config.SEARCH_SIZE)

        # 2. Filter Candidates by Depth
        # Find indices where atlas depth is within tolerance of query depth
        depth_diff = np.abs(self.atlas_depths - depth)
        valid_mask = depth_diff <= self.depth_tolerance

        candidate_indices = self.atlas_indices[valid_mask]

        # Handle edge case: No candidates found (e.g., extreme depth values or sparse data)
        # Fallback: Expand search to entire dataset or return zeros.
        # Here we return zeros if absolutely no match, but usually tolerance is wide enough.
        if len(candidate_indices) == 0:
            return np.zeros(
                (
                    self.config.IMG_SIZE[0],
                    self.config.IMG_SIZE[1],
                    len(self.config.CLASSES),
                ),
                dtype=np.uint8,
            )

        # 3. Similarity Search
        # Get vectors for candidates
        candidate_vectors = self.atlas_vectors[candidate_indices]

        # Compute Euclidean distance
        # (N_candidates, D) - (D,) -> (N_candidates, D) -> norm -> (N_candidates,)
        dists = np.linalg.norm(candidate_vectors - query_vec, axis=1)

        # 4. Retrieve Top-K Neighbors
        # We need the smallest distances
        k = min(self.k_neighbors, len(candidate_indices))

        # argpartition puts the k smallest elements at the front (not necessarily sorted)
        # We use argsort for exact sorting to get the absolute best matches
        sorted_local_indices = np.argsort(dists)[:k]

        # Map back to global atlas indices
        top_k_global_indices = candidate_indices[sorted_local_indices]

        # 5. Label Fusion (Majority Voting)
        # Retrieve masks: (K, H, W, C)
        retrieved_masks = self.atlas_masks[top_k_global_indices]

        # Compute mean across the K neighbors (axis 0)
        # Result shape: (H, W, C) with values in [0, 1]
        vote_map = np.mean(retrieved_masks, axis=0)

        # Threshold to get binary prediction
        prediction = (vote_map >= self.fusion_threshold).astype(np.uint8)

        return prediction
