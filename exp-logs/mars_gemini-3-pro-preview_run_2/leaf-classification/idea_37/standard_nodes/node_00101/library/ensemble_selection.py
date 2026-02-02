import os
import numpy as np
import time
from library.config import Config
from library.utils import calculate_log_loss


class GreedyEnsembleSelector:
    """
    Implements Greedy Forward Selection (with replacement) for ensemble optimization.

    This class handles:
    1. Training the library of experts on Phase 1 data.
    2. Generating and caching validation predictions.
    3. Selecting the optimal subset of experts and their weights to minimize Log Loss.
    """

    def __init__(self, max_iterations=100, tolerance=1e-6):
        """
        Args:
            max_iterations (int): Maximum number of experts to add to the ensemble.
            tolerance (float): Minimum improvement required to continue selection.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.selected_experts_ = []  # List of expert IDs
        self.weights_ = {}  # Dictionary mapping expert_id -> weight (count)
        self.best_score_ = float("inf")
        self.trajectory_ = []  # History of scores

    def _get_cache_path(self):
        """Returns the path for caching validation predictions."""
        return os.path.join(Config.WORKING_DIR, "p1_library_val_preds.npy")

    def _get_expert_ids_cache_path(self):
        """Returns the path for caching expert IDs to ensure alignment."""
        return os.path.join(Config.WORKING_DIR, "p1_library_ids.npy")

    def _generate_pool_predictions(
        self, train_data, val_data, experts, load_cached_data=True
    ):
        """
        Trains all experts and generates validation predictions.
        Implements caching to avoid re-training on the same split.

        Args:
            train_data (dict): Phase 1 training data (streams + y).
            val_data (dict): Phase 1 validation data (streams + y).
            experts (list): List of expert dictionaries from ModelFactory.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Mapping of expert_id -> probability matrix (numpy array).
        """
        preds_path = self._get_cache_path()
        ids_path = self._get_expert_ids_cache_path()

        # Extract expert IDs for verification
        current_expert_ids = [e["id"] for e in experts]

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(preds_path) and os.path.exists(ids_path):
            try:
                cached_ids = np.load(ids_path, allow_pickle=True)
                # Verify cache alignment
                if np.array_equal(cached_ids, current_expert_ids):
                    print("Loading cached validation predictions for expert library...")
                    # Load the big array and split it back into a dict
                    # Shape: (n_experts, n_samples, n_classes)
                    all_preds = np.load(preds_path)
                    pred_dict = {
                        eid: all_preds[i] for i, eid in enumerate(current_expert_ids)
                    }
                    return pred_dict
                else:
                    print(
                        "Cached expert IDs do not match current library. Re-computing..."
                    )
            except Exception as e:
                print(f"Error loading cache: {e}. Re-computing...")

        # 2. Compute from Scratch
        print(
            f"Training {len(experts)} experts and generating validation predictions..."
        )
        pred_dict = {}

        # Pre-allocate list for caching later
        preds_list = []

        for i, expert in enumerate(experts):
            eid = expert["id"]
            model = expert["model"]
            stream_name = expert["stream"]

            print(f"[{i+1}/{len(experts)}] Processing {eid} on {stream_name}...")

            # Select correct stream
            X_train = train_data[stream_name]
            y_train = train_data["y"]
            X_val = val_data[stream_name]

            # Fit
            model.fit(X_train, y_train)

            # Predict
            # Ensure float64 precision
            probs = model.predict_proba(X_val).astype(Config.NP_DTYPE)
            pred_dict[eid] = probs
            preds_list.append(probs)

        # 3. Save to Cache
        try:
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            # Stack into a single array for storage: (n_experts, n_samples, n_classes)
            all_preds = np.stack(preds_list, axis=0)
            np.save(preds_path, all_preds)
            np.save(ids_path, np.array(current_expert_ids))
            print("Validation predictions cached.")
        except Exception as e:
            print(f"Warning: Failed to cache predictions: {e}")

        return pred_dict

    def fit(self, train_data, val_data, experts, load_cached_data=True):
        """
        Runs the Greedy Forward Selection algorithm.

        Args:
            train_data (dict): Phase 1 training data.
            val_data (dict): Phase 1 validation data.
            experts (list): List of expert definitions.
            load_cached_data (bool): Whether to use cached predictions.

        Returns:
            self
        """
        # 1. Get Predictions
        pool_preds = self._generate_pool_predictions(
            train_data, val_data, experts, load_cached_data=load_cached_data
        )

        y_true = val_data["y"]
        expert_ids = [e["id"] for e in experts]

        # 2. Initialize Selection Variables
        # We start with an empty ensemble.
        # To handle the first iteration cleanly, we treat the current ensemble prediction
        # as zeros (or uniform, but zeros works if we track sum).

        # current_sum_probs: Accumulator for the probabilities of the selected experts
        n_samples = len(y_true)
        n_classes = len(
            np.unique(y_true)
        )  # Assuming y_true covers all classes or derived from metadata
        # Ideally get n_classes from the shape of predictions
        first_pred = list(pool_preds.values())[0]
        n_classes = first_pred.shape[1]

        current_sum_probs = np.zeros((n_samples, n_classes), dtype=Config.NP_DTYPE)
        current_size = 0

        best_log_loss = float("inf")

        selected_indices = []  # List of expert IDs added in order

        print(
            f"\nStarting Greedy Forward Selection (Max Iters: {self.max_iterations})..."
        )
        start_time = time.time()

        # 3. Selection Loop
        for iteration in range(self.max_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert_id = None

            # Try adding each expert to the current ensemble
            for eid in expert_ids:
                # Trial ensemble: (current_sum + new_pred) / (current_size + 1)
                # Note: We do this calculation explicitly to maintain precision
                trial_sum = current_sum_probs + pool_preds[eid]
                trial_avg = trial_sum / (current_size + 1)

                score = calculate_log_loss(y_true, trial_avg)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert_id = eid

            # Check for improvement
            # For the first iteration, we always accept (unless score is somehow inf)
            if (
                current_size == 0
                or iteration_best_score < best_log_loss - self.tolerance
            ):
                best_log_loss = iteration_best_score
                current_sum_probs += pool_preds[iteration_best_expert_id]
                current_size += 1
                selected_indices.append(iteration_best_expert_id)
                self.trajectory_.append(best_log_loss)

                print(
                    f"Iter {iteration+1}: Added {iteration_best_expert_id}, "
                    f"Score: {best_log_loss:.15f}"
                )
            else:
                print(
                    f"Iter {iteration+1}: No improvement > {self.tolerance}. Stopping."
                )
                break

        elapsed = time.time() - start_time
        print(f"Selection complete in {elapsed:.2f}s.")
        print(f"Final Ensemble Size: {len(selected_indices)}")
        print(f"Best Validation Log Loss: {best_log_loss:.15f}")

        # 4. Store Results
        self.selected_experts_ = selected_indices
        self.best_score_ = best_log_loss

        # Compute weights (counts)
        self.weights_ = {}
        for eid in selected_indices:
            self.weights_[eid] = self.weights_.get(eid, 0) + 1

        print("Ensemble Weights:")
        for eid, w in self.weights_.items():
            print(f"  - {eid}: {w}")

        return self

    def get_selected_config(self):
        """
        Returns the configuration needed for Phase 2.

        Returns:
            dict: {expert_id: weight}
        """
        return self.weights_
