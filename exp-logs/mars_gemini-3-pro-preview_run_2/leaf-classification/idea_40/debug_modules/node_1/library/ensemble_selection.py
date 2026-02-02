import os
import numpy as np
from collections import Counter

from library.config import (
    TOPOLOGIES,
    SHRINKAGE_GRID,
    VIEWS,
    WORKING_DIR,
    FLOAT_PRECISION,
    RANDOM_SEED,
)
from library.utils import calculate_log_loss, set_seed
from library.data_factory import DataFactory
from library.transformations import MarginalTopology, SpectralTopology, RankTopology
from library.model_library import LDAExpert


class ExpertLibrary:
    """
    Manages the creation, training, and prediction of the library of experts.
    Handles the Cartesian product of Topologies x Shrinkage Estimators x Feature Views.
    """

    def __init__(self, data_factory):
        """
        Args:
            data_factory (DataFactory): Instance for data retrieval.
        """
        self.data_factory = data_factory
        # Map configuration strings to actual classes
        self.topology_map = {
            "marginal": MarginalTopology,
            "spectral": SpectralTopology,
            "rank": RankTopology,
        }
        self.cache_path = os.path.join(WORKING_DIR, "val_predictions.npz")

    def _get_expert_key(self, topology, shrinkage, view):
        """Generates a unique string key for an expert configuration."""
        return f"{topology}___{shrinkage}___{view}"

    def generate_val_predictions(self, load_cached_data=True):
        """
        Generates or loads validation predictions for all expert combinations.

        This method iterates through the grid defined in library.config, trains
        each expert on the training set, and generates probabilities for the
        validation set. Results are cached to disk.

        Args:
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            tuple:
                - preds_dict (dict): Mapping of expert_key -> prediction_matrix (N_val, N_classes)
                - y_val (np.ndarray): Ground truth labels for the validation set.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading validation predictions from {self.cache_path}")
            try:
                with np.load(self.cache_path, allow_pickle=True) as data:
                    # Extract predictions (exclude y_val which is stored separately in the npz)
                    preds_dict = {
                        key: data[key].astype(FLOAT_PRECISION)
                        for key in data.files
                        if key != "y_val"
                    }
                    y_val = data["y_val"]
                return preds_dict, y_val
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing.")

        print("Computing validation predictions for expert library...")
        preds_dict = {}
        y_val_ref = None

        # 2. Iterate through all combinations
        # Grid: Topologies x Shrinkage x Views

        for view in VIEWS:
            # Load data for this view
            X_train, y_train = self.data_factory.get_data("train", view)
            X_val, y_val = self.data_factory.get_data("val", view)

            # Ensure y_val consistency across views (sanity check)
            if y_val_ref is None:
                y_val_ref = y_val
            else:
                if not np.array_equal(y_val, y_val_ref):
                    raise ValueError(
                        "y_val mismatch between views. Data alignment error."
                    )

            for topo_name in TOPOLOGIES.keys():
                # Initialize Topology Transformer
                # We instantiate a new transformer for each view/topology combo
                TransformerClass = self.topology_map[topo_name]
                transformer = TransformerClass()

                # Fit transformer on training data
                # Note: fit_transform might be more efficient if implemented, but we need the fitted object
                transformer.fit(X_train, y_train)

                # Transform both train and validation sets
                X_train_trans = transformer.transform(X_train)
                X_val_trans = transformer.transform(X_val)

                for shrinkage in SHRINKAGE_GRID:
                    key = self._get_expert_key(topo_name, shrinkage, view)

                    # Initialize and Fit LDA Expert
                    lda = LDAExpert(shrinkage=shrinkage)
                    lda.fit(X_train_trans, y_train)

                    # Predict on Validation set
                    preds = lda.predict_proba(X_val_trans)
                    preds_dict[key] = preds

        # 3. Save to cache
        try:
            # Save predictions and targets in a compressed npz file
            np.savez_compressed(self.cache_path, y_val=y_val_ref, **preds_dict)
            print(f"Saved validation predictions to {self.cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return preds_dict, y_val_ref


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement (Caruana et al., 2004).

    This algorithm iteratively adds the model to the ensemble that maximizes
    the ensemble's performance (minimizes log loss) on the validation set.
    Selection with replacement allows weighting models (e.g., selecting a model
    twice gives it double the weight).
    """

    def __init__(self, max_iterations=100, tolerance=1e-6):
        """
        Args:
            max_iterations (int): Maximum number of models to add to the ensemble.
            tolerance (float): Minimum improvement required to continue adding models.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.selected_experts = []  # List of keys (allows duplicates for weighting)
        self.best_score = float("inf")
        self.history = []

    def fit(self, preds_dict, y_true):
        """
        Runs the greedy selection process.

        Args:
            preds_dict (dict): Dictionary mapping expert keys to probability matrices.
            y_true (np.ndarray): Ground truth labels.

        Returns:
            self
        """
        set_seed(RANDOM_SEED)

        expert_keys = list(preds_dict.keys())
        # Ensure precision
        y_true = np.array(y_true, dtype=int)

        # Get shape from first prediction matrix
        n_samples, n_classes = list(preds_dict.values())[0].shape

        # Current ensemble sum of probabilities (unscaled)
        # We maintain the sum to avoid recomputing the average from scratch at every step
        current_ensemble_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        print(
            f"Starting Greedy Forward Selection (Max Iters: {self.max_iterations})..."
        )

        for i in range(self.max_iterations):
            best_iter_score = float("inf")
            best_expert_key = None

            # Try adding each expert to the current ensemble
            for key in expert_keys:
                candidate_preds = preds_dict[key]

                # Calculate new average if we add this expert
                # New Avg = (Current Sum + Candidate) / (Current Count + 1)
                # i is the current number of experts in the ensemble (0-indexed loop)
                new_ensemble_avg = (current_ensemble_sum + candidate_preds) / (i + 1)

                score = calculate_log_loss(y_true, new_ensemble_avg)

                if score < best_iter_score:
                    best_iter_score = score
                    best_expert_key = key

            # Check for improvement
            # For the first iteration (i=0), we always accept the best single model
            # For subsequent iterations, we check against tolerance
            improvement = self.best_score - best_iter_score

            if i == 0 or improvement > self.tolerance:
                self.selected_experts.append(best_expert_key)
                self.best_score = best_iter_score
                current_ensemble_sum += preds_dict[best_expert_key]
                self.history.append((i + 1, best_iter_score, best_expert_key))
                print(
                    f"Iter {i+1}: Added {best_expert_key}, Score: {best_iter_score:.15f}"
                )
            else:
                print(
                    f"Iter {i+1}: No significant improvement ({improvement:.15f} <= {self.tolerance}). Stopping."
                )
                break

        return self

    def get_best_ensemble(self):
        """
        Returns the composition of the best ensemble found.

        Returns:
            list: List of tuples (expert_key, count/weight).
        """
        counts = Counter(self.selected_experts)
        return list(counts.items())
