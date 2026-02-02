import numpy as np
from library.utils import clipped_log_loss, to_float64


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to find the optimal
    linear combination of expert probabilities that minimizes log loss.
    """

    def __init__(self, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Args:
            max_iterations (int): Maximum number of experts to select (ensemble size).
            tolerance (float): Minimum improvement in log loss required to continue selection.
            verbose (bool): Whether to print progress logs.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.selected_experts = []  # List of names, allowing duplicates (replacement)
        self.weights = {}  # Name -> Count
        self.best_score = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values
                                     are probability matrices (N_samples, N_classes).
            y_true (array-like): Ground truth labels (N_samples,).
        """
        # Ensure inputs are float64 for precision
        expert_names = list(predictions_dict.keys())
        expert_preds = {k: to_float64(v) for k, v in predictions_dict.items()}
        y_true = np.array(y_true)

        # Initialize
        self.selected_experts = []
        self.weights = {}
        current_sum_probs = None  # Will hold sum of selected predictions
        current_k = 0  # Number of selected experts

        # Initial best score (infinity)
        self.best_score = float("inf")

        for i in range(self.max_iterations):
            best_iter_score = float("inf")
            best_expert_name = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                pred = expert_preds[name]

                if current_k == 0:
                    # First iteration: candidate is just this expert
                    candidate_probs = pred
                else:
                    # Update mean: (Sum + New) / (k + 1)
                    # We compute (current_sum_probs + pred) / (current_k + 1)
                    candidate_probs = (current_sum_probs + pred) / (current_k + 1)

                score = clipped_log_loss(y_true, candidate_probs)

                if score < best_iter_score:
                    best_iter_score = score
                    best_expert_name = name

            # Check for improvement
            improvement = self.best_score - best_iter_score

            if improvement > self.tolerance:
                self.best_score = best_iter_score
                self.selected_experts.append(best_expert_name)

                # Update running sum
                if current_sum_probs is None:
                    current_sum_probs = expert_preds[best_expert_name]
                else:
                    current_sum_probs += expert_preds[best_expert_name]

                current_k += 1

                if self.verbose:
                    print(
                        f"Iter {i+1}/{self.max_iterations}: Selected '{best_expert_name}' "
                        f"- Score: {self.best_score:.15f} - Improvement: {improvement:.15f}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Iter {i+1}: No sufficient improvement ({improvement:.15f} <= {self.tolerance}). Stopping."
                    )
                break

        # Calculate final weights
        self._calculate_weights()
        if self.verbose:
            print(f"Final Ensemble Weights: {self.weights}")

    def _calculate_weights(self):
        """Helper to convert list of selected experts to weight dictionary."""
        self.weights = {}
        for name in self.selected_experts:
            self.weights[name] = self.weights.get(name, 0) + 1

    def predict(self, predictions_dict):
        """
        Aggregates predictions using the learned weights.

        Args:
            predictions_dict (dict): Dictionary of expert predictions (test set).

        Returns:
            np.ndarray: Weighted average probability matrix.
        """
        if not self.weights:
            raise ValueError(
                "Selector has not been fitted or no experts were selected."
            )

        # Ensure float64
        expert_preds = {k: to_float64(v) for k, v in predictions_dict.items()}

        # Get shape from one of the predictions
        first_key = next(iter(self.weights.keys()))
        n_samples, n_classes = expert_preds[first_key].shape

        weighted_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        total_weight = 0

        for name, weight in self.weights.items():
            if name not in expert_preds:
                print(
                    f"Warning: Selected expert '{name}' not found in predictions dictionary. Skipping."
                )
                continue

            weighted_sum += expert_preds[name] * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight is zero. Cannot make predictions.")

        return weighted_sum / total_weight
