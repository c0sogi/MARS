import numpy as np
from sklearn.metrics import log_loss
from library import config


class GreedySelector:
    """
    Implements Greedy Forward Selection for ensemble optimization.
    Selects experts from a library to maximize validation Log Loss.
    """

    def __init__(self, max_steps=50, tolerance=1e-6):
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.selected_experts = []
        self.best_score = float("inf")
        self.classes_ = None

    def _score(self, y_true, y_pred_proba):
        """
        Calculates Multi-class Log Loss with specific clipping and normalization
        as defined in the task metric.
        """
        # 1. Normalize (as per task description: "rescaled prior to being scored")
        row_sums = y_pred_proba.sum(axis=1)
        # Handle potential zero sums (though unlikely with proper soft probas)
        row_sums[row_sums == 0] = 1.0
        y_pred_norm = y_pred_proba / row_sums[:, np.newaxis]

        # 2. Clip (as per task description)
        y_pred_clipped = np.clip(y_pred_norm, 1e-15, 1 - 1e-15)

        # 3. Calculate Log Loss
        return log_loss(y_true, y_pred_clipped, labels=self.classes_)

    def fit(self, expert_preds_dict, y_true, classes):
        """
        Fits the ensemble selection using Greedy Forward Selection.

        Args:
            expert_preds_dict (dict): Dictionary mapping expert names to prediction arrays (n_samples, n_classes).
            y_true (np.array): True labels for the validation set.
            classes (list): List of class names corresponding to the columns of the prediction arrays.

        Returns:
            self
        """
        self.classes_ = classes
        self.selected_experts = []

        # Ensure predictions are in double precision
        for k, v in expert_preds_dict.items():
            if v.dtype != config.FLOAT_PRECISION:
                expert_preds_dict[k] = v.astype(config.FLOAT_PRECISION)

        # Get dimensions
        first_key = next(iter(expert_preds_dict))
        n_samples, n_classes = expert_preds_dict[first_key].shape

        # Initialize ensemble sum (unweighted)
        current_sum = np.zeros((n_samples, n_classes), dtype=config.FLOAT_PRECISION)

        best_score = float("inf")

        print(f"Starting Greedy Selection with {len(expert_preds_dict)} experts...")
        print(f"Validation Set Size: {n_samples}")

        for step in range(1, self.max_steps + 1):
            step_best_score = float("inf")
            step_best_expert = None

            # Try adding each expert (with replacement)
            for expert_name, preds in expert_preds_dict.items():
                # Calculate potential new ensemble average
                # (current_sum + new_pred) / (current_count + 1)
                trial_sum = current_sum + preds
                trial_avg = trial_sum / step

                score = self._score(y_true, trial_avg)

                if score < step_best_score:
                    step_best_score = score
                    step_best_expert = expert_name

            # Check for improvement
            # Always accept the first expert
            improvement = best_score - step_best_score

            if step == 1 or improvement > self.tolerance:
                best_score = step_best_score
                self.selected_experts.append(step_best_expert)
                current_sum += expert_preds_dict[step_best_expert]

                print(f"Step {step}: Added {step_best_expert}")
                print(f"  Validation Log Loss: {best_score:.15f}")
            else:
                print(
                    f"Step {step}: No significant improvement (Best candidate: {step_best_expert}, Score: {step_best_score:.15f}). Stopping."
                )
                break

        self.best_score = best_score
        print(f"Selection Complete. Selected {len(self.selected_experts)} experts.")
        return self

    def predict(self, expert_preds_dict):
        """
        Computes the weighted average predictions using the selected experts.

        Args:
            expert_preds_dict (dict): Dictionary of predictions (e.g., for test set).

        Returns:
            np.array: The aggregated probability matrix (normalized and clipped).
        """
        if not self.selected_experts:
            raise ValueError("Selector not fitted. Call fit() first.")

        # Get dimensions
        first_key = next(iter(expert_preds_dict))
        n_samples, n_classes = expert_preds_dict[first_key].shape

        final_sum = np.zeros((n_samples, n_classes), dtype=config.FLOAT_PRECISION)

        # Sum predictions based on selection (which includes duplicates/weights)
        for expert_name in self.selected_experts:
            if expert_name not in expert_preds_dict:
                raise KeyError(
                    f"Selected expert '{expert_name}' not found in provided predictions."
                )

            preds = expert_preds_dict[expert_name]
            if preds.dtype != config.FLOAT_PRECISION:
                preds = preds.astype(config.FLOAT_PRECISION)

            final_sum += preds

        # Average
        final_avg = final_sum / len(self.selected_experts)

        # Normalize and Clip (consistent with _score and task requirements)
        row_sums = final_avg.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        final_norm = final_avg / row_sums[:, np.newaxis]

        final_clipped = np.clip(final_norm, 1e-15, 1 - 1e-15)

        return final_clipped
