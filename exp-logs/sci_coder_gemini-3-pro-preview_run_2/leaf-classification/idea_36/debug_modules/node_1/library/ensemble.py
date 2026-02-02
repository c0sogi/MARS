import numpy as np
from collections import Counter
from library.config import SELECTION_CONFIG, FLOAT_PRECISION
from library.utils import clipped_log_loss


class GreedyEnsembleSelector:
    """
    Implements Greedy Forward Selection with Replacement for ensemble optimization.
    Selects experts that maximize the improvement in validation Log Loss.
    """

    def __init__(self):
        self.max_experts = SELECTION_CONFIG["max_experts"]
        self.tolerance = SELECTION_CONFIG["tolerance"]
        self.replacement = SELECTION_CONFIG["replacement"]

        self.selected_experts = []  # List of names in order of selection
        self.best_score = float("inf")
        self.trajectory = []  # To store score history (step, score, expert_name)

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): Dictionary {expert_name: proba_matrix (N, C)}
                                     All matrices must be FLOAT_PRECISION.
            y_true (array-like): Ground truth labels.

        Returns:
            self
        """
        # Ensure input precision and get list of available experts
        expert_names = list(predictions_dict.keys())

        # Initialize ensemble state
        # We maintain the sum of predictions of selected experts to quickly compute averages
        # Shape: (N_samples, N_classes)
        current_sum_preds = None
        current_k = 0

        # Reset state
        self.selected_experts = []
        self.best_score = float("inf")
        self.trajectory = []

        print(
            f"Starting Greedy Forward Selection (Max Experts: {self.max_experts}, Tol: {self.tolerance})..."
        )

        for step in range(self.max_experts):
            best_step_expert = None
            best_step_score = float("inf")

            # Identify candidates
            # If replacement is True, all experts are candidates.
            # If False, only those not yet selected.
            if self.replacement:
                candidates = expert_names
            else:
                candidates = [n for n in expert_names if n not in self.selected_experts]

            if not candidates:
                break

            # Iterate through candidates to find the best addition
            for name in candidates:
                preds = predictions_dict[name].astype(FLOAT_PRECISION)

                if current_k == 0:
                    # First selection: Ensemble is just this expert
                    trial_preds = preds
                else:
                    # Weighted average: (Sum_current + New) / (k + 1)
                    trial_preds = (current_sum_preds + preds) / (current_k + 1)

                score = clipped_log_loss(y_true, trial_preds)

                if score < best_step_score:
                    best_step_score = score
                    best_step_expert = name

            # Evaluate improvement
            improvement = self.best_score - best_step_score

            # Acceptance logic
            # Accept if it's the first expert OR improvement > tolerance
            if current_k == 0 or improvement > self.tolerance:
                self.best_score = best_step_score
                self.selected_experts.append(best_step_expert)
                self.trajectory.append((step + 1, best_step_score, best_step_expert))

                # Update running sum
                best_preds = predictions_dict[best_step_expert].astype(FLOAT_PRECISION)
                if current_k == 0:
                    current_sum_preds = best_preds
                else:
                    current_sum_preds += best_preds

                current_k += 1

                print(
                    f"Step {step+1}: Added {best_step_expert}. New Score: {self.best_score:.15f}. Improvement: {improvement:.15f}"
                )
            else:
                print(
                    f"Step {step+1}: Best candidate {best_step_expert} improved by {improvement:.15f} (Tolerance: {self.tolerance}). Stopping."
                )
                break

        print("Selection Complete.")
        print(f"Final Ensemble Size: {len(self.selected_experts)}")
        print(f"Final Validation Score: {self.best_score:.15f}")

        return self

    def get_weights(self):
        """
        Returns the aggregation weights for the selected experts.
        Since we use simple averaging of the selected set (with duplicates),
        the weight is the count of occurrences divided by total count.

        Returns:
            dict: {expert_name: weight}
        """
        if not self.selected_experts:
            return {}

        counts = Counter(self.selected_experts)
        total = len(self.selected_experts)

        # Normalize to sum to 1
        weights = {name: count / total for name, count in counts.items()}
        return weights

    def predict(self, predictions_dict):
        """
        Computes the ensemble prediction for a new set of expert predictions.
        Uses the weights determined during fitting.

        Args:
            predictions_dict (dict): Dictionary {expert_name: proba_matrix}

        Returns:
            np.array: Aggregated probability matrix.
        """
        if not self.selected_experts:
            raise ValueError("Ensemble is empty. Call fit() first.")

        # Initialize sum
        # Get shape from first expert in input to ensure compatibility
        first_expert_name = list(predictions_dict.keys())[0]
        first_expert_preds = predictions_dict[first_expert_name]
        n_samples, n_classes = first_expert_preds.shape

        ensemble_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        # Sum predictions based on selection counts
        # It's more numerically stable to sum (expert_pred * count) then divide by total
        counts = Counter(self.selected_experts)
        total = len(self.selected_experts)

        for name, count in counts.items():
            if name not in predictions_dict:
                raise KeyError(
                    f"Expert {name} selected but not found in input predictions."
                )

            preds = predictions_dict[name].astype(FLOAT_PRECISION)
            ensemble_sum += preds * count

        ensemble_avg = ensemble_sum / total
        return ensemble_avg
