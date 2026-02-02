import numpy as np
from collections import Counter
from library.utils import clip_log_loss
from library.config import FLOAT_PRECISION


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement for ensemble optimization.

    This selector iteratively adds the expert that maximizes the improvement in
    validation log loss to the ensemble. It allows for weighted ensembling by
    selecting the same expert multiple times.
    """

    def __init__(self, n_iterations=50, tolerance=1e-6, random_state=42):
        """
        Args:
            n_iterations (int): Maximum number of selection steps (ensemble size).
            tolerance (float): Minimum improvement in log loss required to continue.
            random_state (int): Seed for reproducibility (unused in deterministic greedy,
                                but kept for API consistency).
        """
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.selected_experts_ = []
        self.weights_ = {}
        self.best_score_ = float("inf")
        self.trajectory_ = []

    def fit(self, preds_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            preds_dict (dict): Dictionary mapping expert names to probability matrices
                               (numpy arrays of shape (n_samples, n_classes)).
            y_true (array-like): True class labels (n_samples,).

        Returns:
            self: The fitted instance.
        """
        expert_names = list(preds_dict.keys())
        if not expert_names:
            raise ValueError("preds_dict cannot be empty.")

        # Ensure all predictions are float64 for precision
        # and pre-validate shapes
        n_samples = len(y_true)
        clean_preds = {}
        for name, preds in preds_dict.items():
            if preds.shape[0] != n_samples:
                raise ValueError(
                    f"Shape mismatch for expert {name}: {preds.shape} vs labels {n_samples}"
                )
            clean_preds[name] = preds.astype(FLOAT_PRECISION)

        # Initialize state
        # We maintain the sum of probabilities of the currently selected ensemble.
        # clip_log_loss handles normalization (dividing by row sum), so we can work with sums directly.
        current_sum_preds = np.zeros(
            (n_samples, clean_preds[expert_names[0]].shape[1]), dtype=FLOAT_PRECISION
        )

        self.selected_experts_ = []
        self.best_score_ = float("inf")
        self.trajectory_ = []

        print(
            f"Starting Greedy Forward Selection (Max Iterations: {self.n_iterations})..."
        )

        for i in range(self.n_iterations):
            best_iter_score = float("inf")
            best_expert = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                candidate_preds = clean_preds[name]

                # If ensemble is empty, candidate is the ensemble
                if i == 0:
                    temp_sum = candidate_preds
                else:
                    temp_sum = current_sum_preds + candidate_preds

                # Calculate score
                # clip_log_loss normalizes rows, so passing the sum is valid
                score = clip_log_loss(y_true, temp_sum)

                if score < best_iter_score:
                    best_iter_score = score
                    best_expert = name

            # Check for improvement
            # For the first iteration, we accept the best single model regardless of "improvement"
            # (since previous best is inf)
            improvement = self.best_score_ - best_iter_score

            if i == 0 or improvement > self.tolerance:
                self.best_score_ = best_iter_score
                self.selected_experts_.append(best_expert)

                # Update the running sum
                if i == 0:
                    current_sum_preds = clean_preds[best_expert]
                else:
                    current_sum_preds += clean_preds[best_expert]

                self.trajectory_.append((i + 1, best_expert, self.best_score_))
                print(
                    f"Step {i+1}: Added {best_expert}, Validation Log Loss: {self.best_score_:.15f}"
                )
            else:
                print(f"Step {i+1}: No improvement > {self.tolerance}. Stopping.")
                break

        # Calculate final integer weights
        self.weights_ = dict(Counter(self.selected_experts_))

        print("-" * 30)
        print(f"Selection Complete. Final Ensemble Size: {len(self.selected_experts_)}")
        print(f"Final Validation Log Loss: {self.best_score_:.15f}")
        print(f"Selected Experts & Weights: {self.weights_}")

        return self

    def predict_proba(self, preds_dict):
        """
        Computes the weighted average predictions using the fitted weights.

        Args:
            preds_dict (dict): Dictionary mapping expert names to probability matrices.

        Returns:
            np.ndarray: Weighted average probabilities.
        """
        if not self.weights_:
            raise RuntimeError("Selector is not fitted or no experts were selected.")

        n_samples = list(preds_dict.values())[0].shape[0]
        n_classes = list(preds_dict.values())[0].shape[1]

        weighted_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)
        total_weight = 0

        for name, weight in self.weights_.items():
            if name not in preds_dict:
                raise KeyError(
                    f"Expert {name} found in weights but missing from input dictionary."
                )

            weighted_sum += preds_dict[name].astype(FLOAT_PRECISION) * weight
            total_weight += weight

        if total_weight == 0:
            return np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        return weighted_sum / total_weight


def run_selection(preds_dict, y_true, n_iterations=50, tolerance=1e-6):
    """
    Helper function to instantiate and run the selector.
    """
    selector = GreedySelector(n_iterations=n_iterations, tolerance=tolerance)
    selector.fit(preds_dict, y_true)
    return selector.weights_, selector.best_score_
