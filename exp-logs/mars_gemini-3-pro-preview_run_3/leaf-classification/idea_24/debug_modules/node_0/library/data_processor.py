import os
import numpy as np
import library.config as cfg
import library.utils as utils


class CentroidGenerator:
    """
    Handles the geometric transformation and structuring of data for the expert bank.
    Partitions extracted views into orthogonal sets and computes centroids to
    create specialized feature spaces for the Orthogonal-Expert Ensemble.
    """

    def __init__(self):
        self.n_rotations = cfg.N_ROTATIONS
        self.n_experts = cfg.N_EXPERTS
        self.views_per_centroid = cfg.VIEWS_PER_CENTROID
        self.working_dir = cfg.WORKING_DIR

        # Cache file paths for processed centroids
        self.train_centroids_path = os.path.join(
            self.working_dir, "train_centroids.npy"
        )
        self.test_centroids_path = os.path.join(self.working_dir, "test_centroids.npy")

    def compute_orthogonal_centroids(self, img_features: np.ndarray) -> np.ndarray:
        """
        Partitions the 36 extracted views into N_EXPERTS sets of orthogonal views
        and computes their element-wise averages (centroids).

        The logic ensures that each expert sees a set of views that are strictly
        orthogonal (e.g., 0, 90, 180, 270 degrees) to maintain variance alignment.

        Args:
            img_features: Array of shape (N_samples, 36, Feature_Dim)

        Returns:
            centroids: Array of shape (N_samples, N_EXPERTS, Feature_Dim)
        """
        # Calculate stride for orthogonality
        # e.g., 36 rotations / 4 views per expert = stride of 9 (90 degrees)
        stride = self.n_rotations // self.views_per_centroid

        centroids_list = []

        for k in range(self.n_experts):
            # Generate indices for Expert k
            # Expert 0: [0, 9, 18, 27]
            # Expert 1: [1, 10, 19, 28]
            # ...
            indices = [k + i * stride for i in range(self.views_per_centroid)]

            # Extract the specific orthogonal views
            # Shape: (N_samples, Views_Per_Centroid, Feature_Dim)
            views = img_features[:, indices, :]

            # Compute the centroid (mean) across the view dimension
            # Shape: (N_samples, Feature_Dim)
            centroid = np.mean(views, axis=1)

            centroids_list.append(centroid)

        # Stack along the expert dimension
        # Shape: (N_samples, N_EXPERTS, Feature_Dim)
        return np.stack(centroids_list, axis=1)

    def process_features(
        self, raw_data: dict = None, load_cached_data: bool = True
    ) -> dict:
        """
        Orchestrates the centroid generation process with deterministic caching.

        Args:
            raw_data: Dictionary containing 'train_img' and 'test_img' from feature_extractor.
                      Required if cache is missing or load_cached_data is False.
            load_cached_data: Whether to attempt loading from cache.

        Returns:
            Dictionary containing 'train_centroids' and 'test_centroids'.
        """
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(self.train_centroids_path) and os.path.exists(
                self.test_centroids_path
            ):
                print(f"Loading cached centroids from {self.working_dir}...")
                try:
                    train_centroids = np.load(self.train_centroids_path)
                    test_centroids = np.load(self.test_centroids_path)
                    return {
                        "train_centroids": train_centroids,
                        "test_centroids": test_centroids,
                    }
                except Exception as e:
                    print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print("Computing orthogonal centroids...")
        if raw_data is None:
            raise ValueError(
                "Raw data is required when cache is missing or load_cached_data=False."
            )

        if "train_img" not in raw_data or "test_img" not in raw_data:
            raise KeyError("raw_data must contain 'train_img' and 'test_img' keys.")

        train_img = raw_data["train_img"]
        test_img = raw_data["test_img"]

        train_centroids = self.compute_orthogonal_centroids(train_img)
        test_centroids = self.compute_orthogonal_centroids(test_img)

        # 3. Save to cache
        print(f"Saving centroids to {self.working_dir}...")
        np.save(self.train_centroids_path, train_centroids)
        np.save(self.test_centroids_path, test_centroids)

        return {"train_centroids": train_centroids, "test_centroids": test_centroids}

    def prepare_expert_dataset(
        self, centroids: np.ndarray, tab_features: np.ndarray, expert_idx: int
    ) -> np.ndarray:
        """
        Constructs the feature matrix for a specific expert by concatenating
        the expert-specific visual centroid with the raw tabular features.

        This creates the vector [DINO_centroid | ConvNeXt_centroid | Tabular]
        that will be fed into the Selective Feature Topology pipeline.

        Args:
            centroids: Array of shape (N_samples, N_EXPERTS, Vis_Dim)
            tab_features: Array of shape (N_samples, Tab_Dim)
            expert_idx: Index of the expert (0 to N_EXPERTS-1)

        Returns:
            X: Concatenated feature matrix of shape (N_samples, Vis_Dim + Tab_Dim)
        """
        if expert_idx < 0 or expert_idx >= self.n_experts:
            raise ValueError(
                f"Expert index {expert_idx} out of range (0-{self.n_experts-1})"
            )

        # Extract the specific centroid for this expert
        # Shape: (N_samples, Vis_Dim)
        expert_vis_features = centroids[:, expert_idx, :]

        # Concatenate with tabular features
        # Shape: (N_samples, Vis_Dim + Tab_Dim)
        X = np.concatenate([expert_vis_features, tab_features], axis=1)

        return X
