import numpy as np
from collections import Counter
from library.utils import clipped_log_loss
from library.config import FLOAT_PRECISION


class HillClimbingOptimizer:
    """
    Implements Greedy Forward Selection with Replacement (Hill Climbing) to optimize
    ensemble weights for multi-class classification.
    """

    def __init__(self, n_iterations=100, stop_epsilon=1e-6, verbose=True):
        """
        Args:
            n_iterations (int): Maximum number of iterations (additions to ensemble).
            stop_epsilon (float): Minimum improvement required to continue.
            verbose (bool): Whether to print progress.
        """
        self.n_iterations = n_iterations
        self.stop_epsilon = stop_epsilon
        self.verbose = verbose
        self.selected_experts = []  # List of expert names (can contain duplicates)
        self.best_score = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using the validation set.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values
                                     are (N_samples, N_classes) probability arrays.
            y_true (array-like): Ground truth labels (N_samples,).

        Returns:
            list: The list of selected expert names (in order of addition).
        """
        available_experts = list(predictions_dict.keys())

        if not available_experts:
            raise ValueError("predictions_dict cannot be empty.")

        # Ensure we have consistent shapes
        first_key = available_experts[0]
        n_samples, n_classes = predictions_dict[first_key].shape

        # Initialize current ensemble state
        self.selected_experts = []
        self.best_score = float("inf")

        # We maintain the sum of probabilities of the current ensemble
        # to avoid re-summing O(N^2) times.
        # Prediction = current_sum / len(selected_experts)
        current_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        for i in range(self.n_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each available expert to the current ensemble
            for name in available_experts:
                pred = predictions_dict[name]

                # Calculate what the prediction would be if we added this expert
                # New Ensemble Size = Current Size + 1
                k = len(self.selected_experts) + 1

                # Trial prediction: (current_sum + new_pred) / k
                # We do this calculation carefully to preserve precision
                trial_sum = current_sum + pred.astype(FLOAT_PRECISION)
                trial_pred = trial_sum / k

                # Calculate metric
                score = clipped_log_loss(y_true, trial_pred)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = name

            # Check if the best addition improves the score significantly
            # For the first iteration, self.best_score is inf, so it will always improve.
            if iteration_best_score < (self.best_score - self.stop_epsilon):
                self.best_score = iteration_best_score
                self.selected_experts.append(iteration_best_expert)

                # Update the running sum
                current_sum += predictions_dict[iteration_best_expert].astype(
                    FLOAT_PRECISION
                )

                if self.verbose:
                    print(
                        f"Ensemble Selection Iteration {i+1}: "
                        f"Added '{iteration_best_expert}'. "
                        f"Val Log Loss: {self.best_score:.16f}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Ensemble Selection: No improvement at iteration {i+1}. Stopping."
                    )
                break

        return self.selected_experts

    def predict(self, predictions_dict):
        """
        Generates predictions using the fitted ensemble.

        Args:
            predictions_dict (dict): Dictionary of expert predictions.

        Returns:
            np.ndarray: Weighted average probabilities (N_samples, N_classes).
        """
        if not self.selected_experts:
            raise ValueError(
                "Optimizer has not been fitted or no experts were selected."
            )

        first_key = self.selected_experts[0]
        if first_key not in predictions_dict:
            raise KeyError(
                f"Selected expert '{first_key}' not found in predictions_dict."
            )

        n_samples, n_classes = predictions_dict[first_key].shape
        final_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        for name in self.selected_experts:
            if name not in predictions_dict:
                raise KeyError(
                    f"Selected expert '{name}' not found in predictions_dict."
                )
            final_sum += predictions_dict[name].astype(FLOAT_PRECISION)

        return final_sum / len(self.selected_experts)

    def get_weights(self):
        """
        Returns the calculated weights for each expert.

        Returns:
            dict: Dictionary mapping expert names to their relative weight (0-1).
        """
        if not self.selected_experts:
            return {}

        counts = Counter(self.selected_experts)
        total = len(self.selected_experts)
        return {k: v / total for k, v in counts.items()}
