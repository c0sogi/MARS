import numpy as np
from collections import Counter
from library.utils import clipped_log_loss


class GreedyEnsemble:
    """
    Implements Greedy Forward Selection (Hill Climbing) for ensemble optimization.
    Selects experts from a library to minimize validation log loss.
    """

    def __init__(self, max_size: int = 50, tol: float = 1e-6):
        """
        Args:
            max_size (int): Maximum number of experts to select (sum of weights).
            tol (float): Minimum improvement required to continue selection.
        """
        self.max_size = max_size
        self.tol = tol
        self.selected_experts = []  # List of names, allowing duplicates (weights)
        self.weights = {}  # Summary of weights {name: count}
        self.best_score = float("inf")

    def fit(self, expert_preds: dict, y_true: np.ndarray):
        """
        Fits the ensemble by iteratively adding the expert that maximizes metric improvement.

        Args:
            expert_preds (dict): Dictionary {expert_name: probability_matrix}.
                                 Each matrix has shape (n_samples, n_classes).
            y_true (np.ndarray): Ground truth labels.
        """
        available_experts = list(expert_preds.keys())
        if not available_experts:
            raise ValueError("No experts provided in expert_preds dict.")

        # Reset state
        self.selected_experts = []
        self.weights = {}
        self.best_score = float("inf")

        # Accumulator for the sum of predictions of selected experts.
        # We maintain the sum to avoid re-summing the whole list every iteration.
        current_sum_preds = None

        print(f"Starting Greedy Forward Selection (Max Steps: {self.max_size})...")

        for step in range(1, self.max_size + 1):
            best_step_score = float("inf")
            best_expert_name = None

            # Try adding each expert to the current ensemble
            for name in available_experts:
                preds = expert_preds[name].astype(np.float64)

                if current_sum_preds is None:
                    # First selection: Ensemble is just this expert
                    candidate_ensemble = preds
                else:
                    # Current average = (Sum + New) / (Count + 1)
                    # We pass the average to the metric function
                    candidate_ensemble = (current_sum_preds + preds) / step

                # Calculate score
                # Note: clipped_log_loss handles row normalization internally,
                # but passing a valid probability matrix (sum~1) is best practice.
                score = clipped_log_loss(y_true, candidate_ensemble)

                if score < best_step_score:
                    best_step_score = score
                    best_expert_name = name

            # Evaluate improvement
            # For the first step, we always accept (unless score is infinite/nan)
            improvement = self.best_score - best_step_score

            if step == 1 or improvement > self.tol:
                self.best_score = best_step_score
                self.selected_experts.append(best_expert_name)

                # Update accumulator with the chosen expert's predictions
                if current_sum_preds is None:
                    current_sum_preds = (
                        expert_preds[best_expert_name].astype(np.float64).copy()
                    )
                else:
                    current_sum_preds += expert_preds[best_expert_name].astype(
                        np.float64
                    )

                print(
                    f"Step {step}: Selected '{best_expert_name}'. New Score: {self.best_score:.15f}"
                )
            else:
                print(
                    f"Step {step}: No significant improvement (Best: {best_step_score:.15f}, Previous: {self.best_score:.15f}). Stopping."
                )
                break

        # Calculate final weights
        self.weights = dict(Counter(self.selected_experts))
        print("Selection Complete.")
        print(f"Final Ensemble Weights: {self.weights}")
        return self

    def predict(self, expert_preds: dict) -> np.ndarray:
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            expert_preds (dict): Dictionary {expert_name: probability_matrix} for the test/inference set.

        Returns:
            np.ndarray: Weighted average probability matrix.
        """
        if not self.selected_experts:
            raise ValueError("Ensemble is not fitted. Call fit() first.")

        # Check if all selected experts are present in the input
        # Note: self.weights.keys() are the unique selected experts
        missing = [name for name in self.weights.keys() if name not in expert_preds]
        if missing:
            raise KeyError(f"Missing predictions for selected experts: {missing}")

        final_pred = None
        total_weight = 0

        # Compute weighted sum
        for name, weight in self.weights.items():
            preds = expert_preds[name].astype(np.float64)

            if final_pred is None:
                final_pred = preds * weight
            else:
                final_pred += preds * weight

            total_weight += weight

        # Normalize by total weight to get probabilities
        final_pred /= total_weight

        return final_pred
