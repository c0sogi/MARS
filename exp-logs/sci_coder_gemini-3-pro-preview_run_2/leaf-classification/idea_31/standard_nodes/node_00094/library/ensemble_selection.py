import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library import config


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to optimize
    Multi-class Log Loss for an ensemble of probabilistic experts.
    """

    def __init__(self, max_size=config.MAX_ENSEMBLE_SIZE, tolerance=1e-6):
        self.max_size = max_size
        self.tolerance = tolerance
        self.selected_experts = []  # List of expert names in order of selection
        self.best_score = float("inf")
        self.weights = {}  # Name -> Weight (frequency / total_selected)

    def _score_predictions(self, y_true, y_pred):
        """
        Calculates Multi-class Log Loss with specific clipping and normalization
        as defined in the task metric.
        """
        # 1. Row-wise normalization (rescaling)
        # The competition metric rescales rows to sum to 1
        row_sums = y_pred.sum(axis=1)
        # Avoid division by zero (though unlikely with valid probabilities)
        row_sums[row_sums == 0] = 1.0
        y_pred_norm = y_pred / row_sums[:, np.newaxis]

        # 2. Clipping
        # Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

        # 3. Log Loss
        # sklearn's log_loss handles the string labels in y_true automatically
        # provided y_pred columns align with sorted unique classes.
        # Since all experts are trained on the same target set, this alignment is guaranteed.
        return log_loss(y_true, y_pred_clipped)

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices (N_samples, N_classes).
            y_true (array-like): True labels for the validation set.

        Returns:
            self
        """
        expert_names = list(predictions_dict.keys())
        if not expert_names:
            raise ValueError("predictions_dict cannot be empty.")

        n_samples, n_classes = predictions_dict[expert_names[0]].shape

        # Initialize ensemble sum with zeros
        # We use float64 for accumulation to maintain precision
        current_ensemble_sum = np.zeros(
            (n_samples, n_classes), dtype=config.FLOAT_PRECISION
        )

        self.selected_experts = []
        self.best_score = float("inf")

        print(f"Starting Greedy Forward Selection (Max Size: {self.max_size})...")

        # Iteratively add experts
        for i in range(1, self.max_size + 1):
            best_iter_score = float("inf")
            best_iter_expert = None

            # Try adding each candidate expert to the current ensemble
            for name in expert_names:
                candidate_preds = predictions_dict[name]

                # Calculate trial ensemble prediction
                # New Average = (Sum of previous selected + New Candidate) / (Current Size)
                trial_sum = current_ensemble_sum + candidate_preds
                trial_pred = trial_sum / i

                score = self._score_predictions(y_true, trial_pred)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            # Check for improvement
            improvement = self.best_score - best_iter_score

            if improvement > self.tolerance:
                self.best_score = best_iter_score
                self.selected_experts.append(best_iter_expert)
                current_ensemble_sum += predictions_dict[best_iter_expert]
                print(
                    f"Step {i}: Added {best_iter_expert}, Validation Log Loss: {self.best_score:.15f}"
                )
            else:
                print(
                    f"Step {i}: No significant improvement (Best iter: {best_iter_score:.15f} vs Current: {self.best_score:.15f}). Stopping."
                )
                break

        if not self.selected_experts:
            print("Warning: No experts selected. Defaulting to the single best expert.")
            # Fallback: find single best
            best_single_score = float("inf")
            best_single_name = None
            for name in expert_names:
                s = self._score_predictions(y_true, predictions_dict[name])
                if s < best_single_score:
                    best_single_score = s
                    best_single_name = name
            self.selected_experts.append(best_single_name)
            self.best_score = best_single_score

        # Calculate final weights based on selection frequency
        counts = Counter(self.selected_experts)
        total_selected = len(self.selected_experts)
        self.weights = {name: count / total_selected for name, count in counts.items()}

        print("-" * 30)
        print(f"Selection Complete. Ensemble Size: {total_selected}")
        print(f"Final Validation Log Loss: {self.best_score:.15f}")
        print("Selected Experts and Weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        print("-" * 30)

        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to probability matrices.

        Returns:
            np.ndarray: Weighted average probability matrix.
        """
        if not self.weights:
            raise ValueError(
                "Selector has not been fitted or no experts were selected."
            )

        expert_names = list(predictions_dict.keys())
        n_samples, n_classes = predictions_dict[expert_names[0]].shape

        final_pred = np.zeros((n_samples, n_classes), dtype=config.FLOAT_PRECISION)

        for name, weight in self.weights.items():
            if name in predictions_dict:
                final_pred += predictions_dict[name] * weight
            else:
                raise KeyError(
                    f"Selected expert '{name}' not found in provided predictions dictionary."
                )

        return final_pred
