import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library.config import PROB_CLIP


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to optimize
    ensemble weights for Multi-class Log Loss.
    """

    def __init__(self, n_iterations=100, tolerance=1e-6):
        """
        Args:
            n_iterations (int): Maximum number of selection steps.
            tolerance (float): Minimum improvement required to continue selection.
        """
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.selected_experts = []
        self.weights = {}
        self.best_score = float("inf")

    def _score(self, y_true, y_pred):
        """
        Calculates Multi-class Log Loss with rescaling and clipping.

        Args:
            y_true (np.array): True class labels (integers).
            y_pred (np.array): Predicted probabilities (N, C).

        Returns:
            float: Log loss score.
        """
        # 1. Rescale: Ensure rows sum to 1
        row_sums = y_pred.sum(axis=1, keepdims=True)
        # Avoid division by zero (unlikely with valid probabilities)
        row_sums[row_sums == 0] = 1.0
        y_pred_norm = y_pred / row_sums

        # 2. Score: log_loss handles clipping via eps parameter
        return log_loss(y_true, y_pred_norm, eps=PROB_CLIP)

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy selection process.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices (N, C).
            y_true (np.array): True integer labels (N,).

        Returns:
            self
        """
        expert_names = list(predictions_dict.keys())
        if not expert_names:
            raise ValueError("predictions_dict cannot be empty.")

        # Reset state
        self.selected_experts = []
        self.best_score = float("inf")

        # Accumulator for the sum of predictions of selected experts
        # We maintain the sum to avoid re-calculating the average from scratch every time
        ensemble_sum = None
        ensemble_count = 0

        print(f"Starting Greedy Forward Selection (Max Steps: {self.n_iterations})...")

        for step in range(self.n_iterations):
            step_best_score = float("inf")
            step_best_expert = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                candidate_preds = predictions_dict[name]

                if ensemble_sum is None:
                    # First selection: Average is just the candidate
                    trial_pred = candidate_preds
                else:
                    # New Average = (Current Sum + Candidate) / (Count + 1)
                    trial_pred = (ensemble_sum + candidate_preds) / (ensemble_count + 1)

                score = self._score(y_true, trial_pred)

                if score < step_best_score:
                    step_best_score = score
                    step_best_expert = name

            # Check stopping criterion
            # We continue if it's the first step OR if improvement > tolerance
            if step == 0 or (step_best_score < self.best_score - self.tolerance):
                self.best_score = step_best_score
                self.selected_experts.append(step_best_expert)

                # Update the running sum
                if ensemble_sum is None:
                    ensemble_sum = predictions_dict[step_best_expert]
                else:
                    ensemble_sum += predictions_dict[step_best_expert]
                ensemble_count += 1

                print(
                    f"Step {step+1}: Added '{step_best_expert}'. New Best Score: {self.best_score}"
                )
            else:
                print(
                    f"Step {step+1}: No significant improvement (Best Trial: {step_best_score} vs Current: {self.best_score}). Stopping."
                )
                break

        # Calculate final weights based on selection frequency
        counts = Counter(self.selected_experts)
        total = len(self.selected_experts)
        self.weights = {k: v / total for k, v in counts.items()}

        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the fitted weights.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices.
                                     Must contain all keys present in self.weights.

        Returns:
            np.array: Weighted average probabilities.
        """
        if not self.weights:
            raise RuntimeError("Selector is not fitted or no experts were selected.")

        ensemble_pred = None

        for name, weight in self.weights.items():
            if name not in predictions_dict:
                raise KeyError(f"Expert '{name}' not found in predictions_dict.")

            preds = predictions_dict[name]
            # Accumulate weighted predictions
            weighted_preds = preds * weight

            if ensemble_pred is None:
                ensemble_pred = weighted_preds
            else:
                ensemble_pred += weighted_preds

        # Note: Since weights sum to 1 and preds sum to 1, ensemble_pred sums to 1.
        # However, we can re-normalize to be safe against floating point drift.
        row_sums = ensemble_pred.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return ensemble_pred / row_sums

    def get_selected_experts(self):
        """Returns the list of selected expert names in order of selection."""
        return self.selected_experts

    def get_weights(self):
        """Returns the dictionary of optimal weights."""
        return self.weights
