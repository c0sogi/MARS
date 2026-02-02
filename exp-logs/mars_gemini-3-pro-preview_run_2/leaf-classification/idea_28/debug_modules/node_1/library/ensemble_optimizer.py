import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library.config import RANDOM_SEED


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement for ensemble optimization.

    This algorithm iteratively adds models to the ensemble that maximize the
    improvement in Log Loss on the validation set. It allows for weighted
    averaging by selecting the same model multiple times.
    """

    def __init__(self, max_iterations=100, tolerance=1e-6):
        """
        Args:
            max_iterations (int): Maximum number of models to include in the ensemble.
            tolerance (float): Minimum improvement in Log Loss required to continue adding models.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.selected_experts_ = []
        self.weights_ = {}
        self.best_score_ = float("inf")
        self.fitted_ = False

    def fit(self, expert_preds_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            expert_preds_dict (dict): Dictionary where keys are expert names and
                                      values are probability matrices (np.ndarray).
            y_true (np.ndarray): True target labels (integers).

        Returns:
            self
        """
        # Ensure deterministic order for tie-breaking
        expert_names = sorted(expert_preds_dict.keys())

        # Initialize variables
        current_ensemble_preds_sum = None
        current_size = 0
        best_global_score = float("inf")

        # History for tracking
        self.selected_experts_ = []

        print(
            f"Starting Greedy Forward Selection (Max Iterations: {self.max_iterations})..."
        )

        for i in range(self.max_iterations):
            best_iter_score = float("inf")
            best_iter_expert = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                preds = expert_preds_dict[name]

                if current_ensemble_preds_sum is None:
                    # First iteration: candidate is just this expert
                    candidate_preds = preds
                else:
                    # Subsequent iterations: average of existing sum + new expert
                    # Mean = (Sum + New) / (Count + 1)
                    candidate_preds = (current_ensemble_preds_sum + preds) / (
                        current_size + 1
                    )

                # Calculate Log Loss
                # Note: GenerativeExpert already clips probabilities, but log_loss
                # also handles epsilon. We pass labels to ensure safety.
                score = log_loss(y_true, candidate_preds)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            # Check for improvement
            improvement = best_global_score - best_iter_score

            if improvement > self.tolerance:
                # Update state
                self.selected_experts_.append(best_iter_expert)
                best_global_score = best_iter_score

                # Update running sum
                if current_ensemble_preds_sum is None:
                    current_ensemble_preds_sum = expert_preds_dict[best_iter_expert]
                else:
                    current_ensemble_preds_sum += expert_preds_dict[best_iter_expert]

                current_size += 1

                print(
                    f"Iter {i+1}: Added '{best_iter_expert}'. "
                    f"New Log Loss: {best_global_score:.15f} (Improved by {improvement:.15f})"
                )
            else:
                print(
                    f"Stopping at Iter {i+1}. Improvement {improvement:.15f} < Tolerance {self.tolerance}"
                )
                break

        self.best_score_ = best_global_score
        self.weights_ = dict(Counter(self.selected_experts_))
        self.fitted_ = True

        print("-" * 30)
        print(f"Selection Complete. Ensemble Size: {len(self.selected_experts_)}")
        print(f"Final Validation Log Loss: {self.best_score_:.15f}")
        print("Selected Weights:", self.weights_)
        print("-" * 30)

        return self

    def predict_proba(self, expert_preds_dict):
        """
        Computes the weighted average of predictions based on the selected experts.

        Args:
            expert_preds_dict (dict): Dictionary of expert predictions (e.g., on Test set).

        Returns:
            np.ndarray: Aggregated probability matrix.
        """
        if not self.fitted_:
            raise RuntimeError("The instance is not fitted yet. Call 'fit' first.")

        if not self.selected_experts_:
            raise RuntimeError("No experts were selected during fitting.")

        # Compute weighted sum
        # We iterate through the selected list (which contains duplicates for weighting)
        # to reconstruct the sum.

        # Optimization: Instead of summing N arrays where N is ensemble size,
        # sum unique arrays multiplied by their weights.

        final_preds_sum = None
        total_weight = 0

        for name, weight in self.weights_.items():
            if name not in expert_preds_dict:
                raise KeyError(
                    f"Expert '{name}' selected during fit is missing from prediction dictionary."
                )

            preds = expert_preds_dict[name]
            weighted_preds = preds * weight

            if final_preds_sum is None:
                final_preds_sum = weighted_preds
            else:
                final_preds_sum += weighted_preds

            total_weight += weight

        # Compute mean
        return final_preds_sum / total_weight

    def get_selected_weights(self):
        """
        Returns the dictionary of selected experts and their integer weights.
        """
        return self.weights_
