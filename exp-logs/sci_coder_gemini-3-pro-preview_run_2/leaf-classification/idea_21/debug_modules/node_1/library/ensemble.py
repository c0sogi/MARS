import numpy as np
from sklearn.metrics import log_loss
from library.config import SELECTION_ITERATIONS


class GreedyEnsembleSelector:
    """
    Implements a Greedy Forward Selection strategy to construct an ensemble of experts.
    The selector iteratively adds the expert that maximizes the improvement in
    validation Log Loss.
    """

    def __init__(self, n_iterations=SELECTION_ITERATIONS):
        """
        Initialize the selector.

        Args:
            n_iterations (int): Number of iterations for the greedy selection process.
                                Defaults to SELECTION_ITERATIONS from config.
        """
        self.n_iterations = n_iterations
        self.weights = {}
        self.best_score = float("inf")
        self.history = []

    def _clip_probabilities(self, preds):
        """
        Clips probabilities to the range [1e-15, 1 - 1e-15] to prevent
        extremes in the log function, as per task specifications.

        Args:
            preds (np.array): Probability matrix.

        Returns:
            np.array: Clipped probability matrix.
        """
        return np.clip(preds, 1e-15, 1 - 1e-15)

    def fit(self, expert_preds_dict, y_true, classes=None):
        """
        Fits the ensemble weights using Greedy Forward Selection.

        Args:
            expert_preds_dict (dict): Dictionary mapping expert names (str) to
                                      probability matrices (np.array of shape (n_samples, n_classes)).
            y_true (np.array): True labels for the validation set.
            classes (np.array, optional): List of all unique class labels. Passed to log_loss
                                          to ensure correct handling if y_true is missing some classes.
        """
        # Initialize weights for all experts to 0
        self.weights = {name: 0 for name in expert_preds_dict.keys()}

        # Get dimensions from the first expert's predictions
        expert_names = list(expert_preds_dict.keys())
        if not expert_names:
            raise ValueError("expert_preds_dict cannot be empty.")

        first_preds = expert_preds_dict[expert_names[0]]
        n_samples, n_classes = first_preds.shape

        # Initialize the current ensemble sum (unweighted sum of selected experts)
        current_ensemble_sum = np.zeros((n_samples, n_classes), dtype=np.float64)

        print(f"Starting Greedy Forward Selection (Iterations={self.n_iterations})...")

        for i in range(self.n_iterations):
            best_iter_score = float("inf")
            best_iter_expert = None

            # Try adding each expert to the ensemble
            for name, preds in expert_preds_dict.items():
                # Calculate the new average if we add this expert
                # New Sum = Current Sum + Candidate Preds
                # New Count = i + 1
                temp_sum = current_ensemble_sum + preds
                temp_avg = temp_sum / (i + 1)

                # Clip probabilities before scoring
                temp_avg = self._clip_probabilities(temp_avg)

                # Calculate Log Loss
                try:
                    score = log_loss(y_true, temp_avg, labels=classes)
                except Exception as e:
                    print(f"Error calculating log_loss for expert '{name}': {e}")
                    raise e

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            # Update the ensemble with the best expert from this iteration
            if best_iter_expert is not None:
                current_ensemble_sum += expert_preds_dict[best_iter_expert]
                self.weights[best_iter_expert] += 1
                self.best_score = best_iter_score
                self.history.append((best_iter_expert, best_iter_score))

                print(
                    f"Iteration {i+1}: Added Expert '{best_iter_expert}' | Validation Log Loss: {best_iter_score}"
                )
            else:
                print(f"Iteration {i+1}: No improvement found.")

        print("-" * 30)
        print("Selection Complete.")
        print("Final Ensemble Weights:")
        for name, weight in self.weights.items():
            if weight > 0:
                print(f"  {name}: {weight}")
        print(f"Best Validation Score: {self.best_score}")
        print("-" * 30)

    def predict(self, expert_preds_dict):
        """
        Generates predictions using the fitted ensemble weights.

        Args:
            expert_preds_dict (dict): Dictionary mapping expert names to probability matrices
                                      (usually for the test set).

        Returns:
            np.array: Weighted average probability matrix.
        """
        if not self.weights:
            raise RuntimeError("EnsembleSelector is not fitted. Call fit() first.")

        # Validate input
        first_key = next(iter(expert_preds_dict))
        n_samples, n_classes = expert_preds_dict[first_key].shape

        weighted_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        total_weight = 0.0

        for name, weight in self.weights.items():
            if weight > 0:
                if name not in expert_preds_dict:
                    raise KeyError(
                        f"Expert '{name}' was selected with weight {weight} but is missing from prediction input."
                    )

                weighted_sum += weight * expert_preds_dict[name]
                total_weight += weight

        if total_weight == 0:
            # Fallback: if no weights (shouldn't happen if n_iterations > 0), simple average
            print(
                "Warning: Total weight is 0. Falling back to simple average of all experts."
            )
            for preds in expert_preds_dict.values():
                weighted_sum += preds
            total_weight = len(expert_preds_dict)

        # Compute average
        final_preds = weighted_sum / total_weight

        # Clip probabilities
        final_preds = self._clip_probabilities(final_preds)

        return final_preds
