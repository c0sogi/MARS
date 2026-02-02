import torch
import numpy as np
import pandas as pd
import gc
from typing import Optional, Union, List
from library.config import Config
from library.utils import get_logger

logger = get_logger("knn_embedding")


class KNNFeatureExtractor:
    """
    Implements GPU-accelerated k-Nearest Neighbors feature extraction.
    Generates 'Manifold-Aware' features: Local Density and Local Class Priors.
    """

    def __init__(self, k: int = Config.KNN_K, device: str = "cuda"):
        """
        Args:
            k (int): Number of neighbors to consider.
            device (str): Compute device ('cuda' or 'cpu').
        """
        self.k = k
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Reference data storage
        self.X_ref: Optional[torch.Tensor] = None
        self.y_ref: Optional[torch.Tensor] = None
        self.ref_sq: Optional[torch.Tensor] = None
        self.classes_: Optional[np.ndarray] = None

        logger.info(f"KNNFeatureExtractor initialized with k={k} on {self.device}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Stores reference data on the GPU for querying.

        Args:
            X: Reference features (Training set).
            y: Reference targets (Training labels).
        """
        logger.info("Fitting KNNFeatureExtractor (moving reference data to GPU)...")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Identify unique classes for probability columns
        self.classes_ = np.unique(y)
        self.classes_.sort()
        logger.info(f"Identified {len(self.classes_)} classes: {self.classes_}")

        # Move to device
        # Use float32 for features to save memory/speed, int64 for targets
        self.X_ref = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y_ref = torch.tensor(y, dtype=torch.int64, device=self.device)

        # Precompute squared norms for reference data: ||y||^2
        # Shape: (N_ref, )
        self.ref_sq = (self.X_ref**2).sum(dim=1)

        logger.info(f"Reference data shape: {self.X_ref.shape}")

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: int = 2048,
        exclude_self: bool = False,
    ) -> pd.DataFrame:
        """
        Computes k-NN features for the query set X.

        Args:
            X: Query features.
            batch_size: Batch size for distance computation.
                        Note: Config.KNN_BATCH_SIZE (10000) might be too large for full attention
                        against 3M rows on 40GB GPU. Defaulting to safe 2048.
            exclude_self: If True, excludes the nearest neighbor (assumed to be self).
                          Use True for Training set, False for Val/Test.

        Returns:
            pd.DataFrame: DataFrame containing 'KNN_Density' and 'KNN_Class_{c}_Prob'.
        """
        if self.X_ref is None:
            raise RuntimeError("Model must be fit before transform.")

        # Convert query to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]

        # Prepare output containers
        # Density: Mean distance to neighbors
        density_features = np.zeros((n_samples,), dtype=np.float32)
        # Probabilities: One column per class
        prob_features = np.zeros((n_samples, len(self.classes_)), dtype=np.float32)

        # Determine actual k to fetch
        # If exclude_self is True, we fetch k+1 and drop the first one
        k_fetch = self.k + 1 if exclude_self else self.k

        logger.info(
            f"Starting transform on {n_samples} samples. Batch size: {batch_size}. Exclude self: {exclude_self}"
        )

        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            # --- 1. Prepare Batch ---
            batch_np = X[start_idx:end_idx]
            batch_tensor = torch.tensor(
                batch_np, dtype=torch.float32, device=self.device
            )

            # --- 2. Compute Distances ---
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
            # Shape: (Batch, N_ref)
            batch_sq = (batch_tensor**2).sum(dim=1, keepdim=True)

            # dist_sq = batch_sq + ref_sq - 2 * batch @ ref.T
            # Note: We do this implicitly to save memory if possible, but PyTorch expansion is efficient
            dist_sq = torch.addmm(
                self.ref_sq.unsqueeze(0), batch_tensor, self.X_ref.t(), beta=1, alpha=-2
            )
            dist_sq += batch_sq

            # Numerical stability: clamp negative values to 0
            dist_sq = torch.clamp(dist_sq, min=0.0)

            # --- 3. Find Nearest Neighbors ---
            # We need the smallest distances. topk returns largest, so we use largest=False
            # values: squared distances, indices: neighbor indices
            topk_vals, topk_inds = torch.topk(dist_sq, k=k_fetch, dim=1, largest=False)

            # --- 4. Handle Exclude Self ---
            if exclude_self:
                # Drop the first column (nearest neighbor, self)
                topk_vals = topk_vals[:, 1:]
                topk_inds = topk_inds[:, 1:]

            # --- 5. Compute Features (on GPU) ---

            # Feature A: Density (Mean Euclidean Distance)
            # sqrt of squared distances
            distances = torch.sqrt(topk_vals)
            batch_density = distances.mean(dim=1)

            # Feature B: Class Priors (Soft Voting)
            # Gather labels of neighbors
            # Shape: (Batch, k)
            neighbor_labels = self.y_ref[topk_inds]

            # Compute counts for each class
            # We create a one-hot like encoding and sum
            batch_probs = torch.zeros(
                (end_idx - start_idx, len(self.classes_)),
                device=self.device,
                dtype=torch.float32,
            )

            for i, cls in enumerate(self.classes_):
                # Check equality and sum across neighbors (dim 1)
                # (neighbor_labels == cls) returns boolean tensor
                count = (neighbor_labels == cls).sum(dim=1).float()
                batch_probs[:, i] = count / self.k

            # --- 6. Store Results (CPU) ---
            density_features[start_idx:end_idx] = batch_density.cpu().numpy()
            prob_features[start_idx:end_idx] = batch_probs.cpu().numpy()

            # --- 7. Cleanup ---
            del (
                batch_tensor,
                dist_sq,
                topk_vals,
                topk_inds,
                distances,
                neighbor_labels,
                batch_probs,
            )
            # Occasionally empty cache to prevent fragmentation
            if (start_idx // batch_size) % 10 == 0:
                torch.cuda.empty_cache()

        logger.info("Transform complete.")

        # Construct DataFrame
        feature_dict = {"KNN_Density": density_features}
        for i, cls in enumerate(self.classes_):
            feature_dict[f"KNN_Class_{cls}_Prob"] = prob_features[:, i]

        return pd.DataFrame(feature_dict)

    def clear_memory(self):
        """Releases GPU memory occupied by reference data."""
        self.X_ref = None
        self.y_ref = None
        self.ref_sq = None
        torch.cuda.empty_cache()
        gc.collect()
