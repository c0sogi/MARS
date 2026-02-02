import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library.config import (
    SELECTION_MAX_ITER,
    SELECTION_TOLERANCE,
    CLIP_MIN,
    CLIP_MAX,
)


class GreedySelector:
    """
    Implements Greedy Forward Selection (Hill Climbing) with replacement to find
    the optimal linear combination of models in the library.

    This method iteratively adds the model that minimizes the validation Log Loss
    to the current ensemble. The final weights correspond to the frequency of
    each model's selection.
    """

    def __init__(self, max_iter=SELECTION_MAX_ITER, tol=SELECTION_TOLERANCE):
        """
        Args:
            max_iter (int): Maximum number of models to add to the ensemble.
            tol (float): Minimum improvement in Log Loss required to continue adding models.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.selected_models = []  # List of model names in order of selection
        self.best_score = float("inf")
        self.weights_ = []  # List of (model_name, count) tuples

    def _process_probs(self, probs):
        """
        Applies the competition-specific post-processing:
        1. Rescale rows to sum to 1.
        2. Clip probabilities to avoid log(0).
        """
        # 1. Rescale (Normalize)
        # Add epsilon to sum to avoid division by zero if a row is all zeros (unlikely)
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        normalized = probs / row_sums

        # 2. Clip
        clipped = np.clip(normalized, CLIP_MIN, CLIP_MAX)
        return clipped

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using validation data.

        Args:
            predictions_dict (dict): Dictionary mapping model names to prediction arrays (N_samples, N_classes).
            y_true (array-like): True integer labels for the validation set.

        Returns:
            self
        """
        available_models = list(predictions_dict.keys())
        n_samples = len(y_true)

        # Initialize ensemble sum accumulator
        # We accumulate the sum of probabilities to avoid re-summing the whole history every iteration
        current_ensemble_sum = None
        n_selected = 0

        print(
            f"Starting Greedy Forward Selection (Max Iter: {self.max_iter}, Tol: {self.tol})..."
        )

        for i in range(self.max_iter):
            best_iter_score = float("inf")
            best_model_name = None

            # Try adding each available model to the current ensemble
            for model_name in available_models:
                pred = predictions_dict[model_name]

                # Calculate trial ensemble prediction
                if current_ensemble_sum is None:
                    # First iteration: ensemble is just this model
                    trial_pred = pred
                else:
                    # Current average = (Sum + New_Pred) / (N + 1)
                    # We only need the probabilities to compute loss
                    trial_pred = (current_ensemble_sum + pred) / (n_selected + 1)

                # Post-process for metric calculation
                final_trial_pred = self._process_probs(trial_pred)

                # Calculate Log Loss
                # Cite debug_lesson_7: Explicitly pass labels matching prediction shape to handle sparse validation data
                score = log_loss(
                    y_true,
                    final_trial_pred,
                    labels=np.arange(final_trial_pred.shape[1]),
                )

                if score < best_iter_score:
                    best_iter_score = score
                    best_model_name = model_name

            # Check for improvement
            improvement = self.best_score - best_iter_score

            # Logic for acceptance
            accept_move = False
            if n_selected == 0:
                # Always accept the first model
                accept_move = True
            elif improvement > self.tol:
                # Accept if improvement exceeds tolerance
                accept_move = True

            if accept_move:
                self.best_score = best_iter_score
                self.selected_models.append(best_model_name)

                # Update the running sum
                if current_ensemble_sum is None:
                    current_ensemble_sum = predictions_dict[best_model_name]
                else:
                    current_ensemble_sum += predictions_dict[best_model_name]

                n_selected += 1

                # Verbose logging for tracking
                print(
                    f"Iter {i+1}/{self.max_iter}: Added {best_model_name:<25} | Val LogLoss: {self.best_score:.6f} | Improv: {improvement:.6f}"
                )
            else:
                print(
                    f"Iter {i+1}/{self.max_iter}: No sufficient improvement (Best: {best_iter_score:.6f}, Improv: {improvement:.6f}). Stopping."
                )
                break

        # Calculate final weights
        counts = Counter(self.selected_models)
        self.weights_ = list(counts.items())

        print("-" * 60)
        print("Final Ensemble Weights:")
        for name, count in self.weights_:
            print(f"  - {name}: {count}")
        print("-" * 60)

        return self

    def get_weights(self):
        """Returns the learned weights as a list of (model_name, count) tuples."""
        return self.weights_


def predict_weighted(predictions_dict, weights):
    """
    Computes the weighted average of predictions.

    Args:
        predictions_dict (dict): Dictionary mapping model names to prediction arrays.
        weights (list): List of (model_name, weight) tuples.

    Returns:
        np.ndarray: The final weighted probability matrix (N_samples, N_classes).
    """
    final_pred = None
    total_weight = 0

    for name, weight in weights:
        if name not in predictions_dict:
            # This implies a mismatch between training and inference keys
            print(
                f"Warning: Model {name} found in weights but not in predictions dictionary. Skipping."
            )
            continue

        pred = predictions_dict[name]

        if final_pred is None:
            final_pred = weight * pred
        else:
            final_pred += weight * pred

        total_weight += weight

    if total_weight == 0:
        raise ValueError(
            "Total weight is zero. Ensure weights match the prediction dictionary keys."
        )

    # Compute weighted average
    avg_pred = final_pred / total_weight

    # Apply final competition-spec formatting (Normalize -> Clip)
    # 1. Normalize
    row_sums = avg_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = avg_pred / row_sums

    # 2. Clip
    clipped = np.clip(normalized, CLIP_MIN, CLIP_MAX)

    return clipped
