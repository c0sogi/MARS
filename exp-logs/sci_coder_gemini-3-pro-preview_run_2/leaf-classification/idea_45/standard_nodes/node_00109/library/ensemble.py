import numpy as np
from sklearn.metrics import log_loss
from library.config import (
    SELECTION_MAX_ITER,
    SELECTION_TOLERANCE,
    SELECTION_WITH_REPLACEMENT,
    PROB_CLIP_MIN,
    PROB_CLIP_MAX,
    FLOAT_PRECISION,
)


class GreedySelector:
    """
    Implements Greedy Forward Selection for ensemble construction.
    Iteratively adds models to the ensemble that maximize the improvement in Log Loss.
    """

    def __init__(self):
        self.selected_experts = []
        self.best_loss = float("inf")
        self.history = []

    def _clip_probabilities(self, preds):
        """
        Clips probabilities to the range [1e-15, 1-1e-15] to avoid log loss extremes.
        """
        return np.clip(preds, PROB_CLIP_MIN, PROB_CLIP_MAX)

    def fit(self, predictions_dict, y_true):
        """
        Selects the optimal subset of experts using greedy forward selection.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to prediction matrices
                                     (n_samples, n_classes).
            y_true (np.array): True class labels (n_samples,).

        Returns:
            self
        """
        available_experts = list(predictions_dict.keys())
        n_samples = len(y_true)

        # Initialize current ensemble sum of predictions
        # We maintain the sum to avoid re-summing the entire history every iteration
        # Start with zeros (shape of one prediction matrix)
        sample_key = available_experts[0]
        n_classes = predictions_dict[sample_key].shape[1]
        current_sum_preds = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        print(f"Starting Greedy Selection with {len(available_experts)} candidates...")
        print(f"Max Iterations: {SELECTION_MAX_ITER}, Tolerance: {SELECTION_TOLERANCE}")

        for i in range(SELECTION_MAX_ITER):
            best_iter_loss = float("inf")
            best_iter_expert = None

            # Number of experts currently in the ensemble (before adding candidate)
            current_k = len(self.selected_experts)

            # Try adding each available expert
            for expert_name in available_experts:
                # If replacement is not allowed and expert is already selected, skip
                if (
                    not SELECTION_WITH_REPLACEMENT
                    and expert_name in self.selected_experts
                ):
                    continue

                # Get candidate predictions
                candidate_preds = predictions_dict[expert_name].astype(FLOAT_PRECISION)

                # Calculate temporary ensemble mean
                # New Mean = (Sum of current + Candidate) / (k + 1)
                temp_sum = current_sum_preds + candidate_preds
                temp_mean = temp_sum / (current_k + 1)

                # Clip and Score
                temp_mean_clipped = self._clip_probabilities(temp_mean)
                loss = log_loss(
                    y_true, temp_mean_clipped, labels=list(range(n_classes))
                )

                if loss < best_iter_loss:
                    best_iter_loss = loss
                    best_iter_expert = expert_name

            # Check for improvement
            improvement = self.best_loss - best_iter_loss

            # If first iteration, we accept the best single model regardless of "improvement"
            # (since best_loss is inf)
            if i == 0 or improvement > SELECTION_TOLERANCE:
                self.best_loss = best_iter_loss
                self.selected_experts.append(best_iter_expert)

                # Update the running sum
                current_sum_preds += predictions_dict[best_iter_expert].astype(
                    FLOAT_PRECISION
                )

                self.history.append((i + 1, best_iter_expert, best_iter_loss))
                print(
                    f"Iter {i+1}: Added {best_iter_expert}, Loss: {best_iter_loss:.15f}, Improv: {improvement:.15f}"
                )
            else:
                print(f"Iter {i+1}: No improvement > {SELECTION_TOLERANCE}. Stopping.")
                break

        print(f"Selection Complete. Ensemble size: {len(self.selected_experts)}")
        return self

    def predict(self, predictions_dict):
        """
        Computes the aggregated predictions using the selected experts.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to prediction matrices.

        Returns:
            np.array: Aggregated probability matrix (n_samples, n_classes).
        """
        if not self.selected_experts:
            raise ValueError(
                "Selector has not been fitted or no experts were selected."
            )

        # Initialize sum
        sample_key = self.selected_experts[0]
        sample_pred = predictions_dict[sample_key]
        n_samples, n_classes = sample_pred.shape

        final_sum_preds = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        # Sum predictions from selected experts
        # Note: self.selected_experts may contain duplicates (integer weighting)
        for expert_name in self.selected_experts:
            if expert_name not in predictions_dict:
                raise KeyError(
                    f"Selected expert '{expert_name}' not found in prediction dictionary."
                )

            final_sum_preds += predictions_dict[expert_name].astype(FLOAT_PRECISION)

        # Compute mean
        final_mean_preds = final_sum_preds / len(self.selected_experts)

        # Apply final clipping
        return self._clip_probabilities(final_mean_preds)
