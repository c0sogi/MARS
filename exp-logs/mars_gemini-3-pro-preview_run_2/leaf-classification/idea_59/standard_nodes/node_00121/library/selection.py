import numpy as np
from collections import Counter
from library.library import build_expert_library, train_experts, predict_experts
from library.utils import clipped_log_loss, ensure_float64


def train_and_predict_library(X_train_dict, y_train, X_val_dict, subset_size=None):
    """
    Trains all candidate experts on the training set and generates validation probabilities.

    Args:
        X_train_dict (dict): Dictionary mapping scope names ('Global', 'Physical', 'Factorized')
                             to training feature arrays.
        y_train (np.ndarray): Training labels (encoded).
        X_val_dict (dict): Dictionary mapping scope names to validation feature arrays.
        subset_size (int, optional): If provided, trains on a subset of the data for debugging/testing.

    Returns:
        tuple: (trained_library, val_predictions)
            trained_library (dict): Dictionary of trained expert pipelines.
            val_predictions (dict): Dictionary of validation probability matrices.
    """
    # Handle subsetting for debugging/fast checks
    if subset_size is not None and subset_size > 0:
        # Ensure we don't exceed available data
        limit = min(subset_size, len(y_train))
        print(f"Subsetting training data to {limit} samples.")
        y_train_subset = y_train[:limit]
        X_train_subset_dict = {}
        for scope, X in X_train_dict.items():
            X_train_subset_dict[scope] = X[:limit]
    else:
        y_train_subset = y_train
        X_train_subset_dict = X_train_dict

    # 1. Build Library
    # Constructs the Cartesian product of Scopes x Bases x Shrinkage
    library = build_expert_library()

    # 2. Train Experts
    # Fits all pipelines on the training data
    trained_library = train_experts(library, X_train_subset_dict, y_train_subset)

    # 3. Predict on Validation
    # Generates probability matrices for the validation set
    val_predictions = predict_experts(trained_library, X_val_dict)

    return trained_library, val_predictions


class GreedySelector:
    """
    Performs Greedy Forward Selection with replacement to optimize ensemble weights.
    Iteratively adds the expert that minimizes the validation log loss of the ensemble.
    """

    def __init__(self, max_iter=20):
        """
        Args:
            max_iter (int): Maximum number of experts to add to the ensemble.
        """
        self.max_iter = max_iter
        self.selected_experts = []
        self.best_loss = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices (n_samples, n_classes).
            y_true (np.ndarray): Ground truth labels (encoded, 1D array).

        Returns:
            self
        """
        expert_names = list(predictions_dict.keys())
        if not expert_names:
            raise ValueError("predictions_dict is empty")

        # Get dimensions from the first prediction matrix
        n_samples, n_classes = list(predictions_dict.values())[0].shape

        # Initialize ensemble probability accumulator
        # We maintain the running average probability matrix
        current_ensemble_prob = np.zeros((n_samples, n_classes), dtype=np.float64)
        self.selected_experts = []
        self.best_loss = float("inf")

        print(f"Starting Greedy Forward Selection (Max Iter: {self.max_iter})...")

        for i in range(self.max_iter):
            iteration_best_loss = float("inf")
            iteration_best_expert = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                prob = ensure_float64(predictions_dict[name])

                # Calculate candidate ensemble probability
                # Formula: NewAvg = (OldAvg * i + NewProb) / (i + 1)
                if i == 0:
                    candidate_prob = prob
                else:
                    candidate_prob = (current_ensemble_prob * i + prob) / (i + 1)

                # Evaluate metric
                loss = clipped_log_loss(y_true, candidate_prob)

                if loss < iteration_best_loss:
                    iteration_best_loss = loss
                    iteration_best_expert = name

            # Check for improvement
            # We strictly require improvement to continue, or we could continue until max_iter
            # The strategy here is to continue as long as we find a best candidate for the round,
            # but usually, we track the global best. If the round best is worse than global best,
            # standard forward selection might stop. However, "bagging" often benefits from more estimators.
            # We will adopt the strategy: Always add the best of the round, but track the global minimum.
            # If the best of the round doesn't improve the GLOBAL best, we stop (early stopping).

            if iteration_best_loss < self.best_loss:
                self.best_loss = iteration_best_loss
                self.selected_experts.append(iteration_best_expert)

                # Update current ensemble probability permanently for next iteration
                prob_best = ensure_float64(predictions_dict[iteration_best_expert])
                if i == 0:
                    current_ensemble_prob = prob_best
                else:
                    current_ensemble_prob = (current_ensemble_prob * i + prob_best) / (
                        i + 1
                    )

                print(
                    f"Iter {i+1}: Added {iteration_best_expert}, Val Loss: {self.best_loss:.15f}"
                )
            else:
                print(
                    f"Iter {i+1}: No improvement (Loss: {iteration_best_loss:.15f} >= Best: {self.best_loss:.15f}). Stopping."
                )
                break

        return self

    def get_selected_experts_with_weights(self):
        """
        Returns the unique selected experts and their integer weights (counts).

        Returns:
            tuple: (unique_experts_list, weights_list)
        """
        counts = Counter(self.selected_experts)
        unique_experts = list(counts.keys())
        weights = [counts[e] for e in unique_experts]
        return unique_experts, weights

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary of probability matrices (e.g., for test set).

        Returns:
            np.ndarray: Weighted average probabilities.
        """
        if not self.selected_experts:
            raise ValueError(
                "Selector has not been fitted or no experts were selected."
            )

        n_samples = list(predictions_dict.values())[0].shape[0]
        n_classes = list(predictions_dict.values())[0].shape[1]

        final_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        # Sum probabilities based on selection counts
        # This is equivalent to averaging over the list of selected experts (with repetitions)
        for name in self.selected_experts:
            final_probs += ensure_float64(predictions_dict[name])

        final_probs /= len(self.selected_experts)

        return final_probs
