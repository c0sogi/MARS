import numpy as np
from sklearn.metrics import log_loss


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement for ensemble optimization.
    Iteratively adds experts to the ensemble that maximize the improvement in Log Loss.
    """

    def __init__(self, max_iter=20, tol=1e-6, random_state=42):
        """
        Args:
            max_iter (int): Maximum number of experts to add to the ensemble.
            tol (float): Minimum improvement in Log Loss required to continue adding experts.
            random_state (int): Seed for reproducibility (if needed).
        """
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selected_experts = []
        self.weights = {}
        self.best_score = float("inf")

    def _clip_probabilities(self, preds):
        """
        Clips probabilities to [1e-15, 1-1e-15] to avoid extremes in the log function,
        consistent with the competition metric.

        Args:
            preds (np.array): Probability matrix.

        Returns:
            np.array: Clipped probability matrix.
        """
        clip_val = 1e-15
        return np.clip(preds, clip_val, 1 - clip_val)

    def fit(self, expert_preds_dict, y_true):
        """
        Fits the ensemble weights using Greedy Forward Selection.

        Args:
            expert_preds_dict (dict): Dictionary where keys are expert names and
                                      values are (N_samples, N_classes) probability arrays (float64).
            y_true (np.array): True labels (N_samples,). Can be strings or integers.
        """
        print(
            f"Starting Greedy Forward Selection (Max Iter: {self.max_iter}, Tol: {self.tol})..."
        )

        # Reset state
        self.selected_experts = []
        self.weights = {}
        expert_names = list(expert_preds_dict.keys())

        if not expert_names:
            raise ValueError("No experts provided in expert_preds_dict.")

        # Get dimensions from the first expert
        first_expert_preds = list(expert_preds_dict.values())[0]
        n_samples, n_classes = first_expert_preds.shape

        # Initialize current ensemble sum as zeros
        current_sum_preds = np.zeros((n_samples, n_classes), dtype=np.float64)
        current_size = 0

        # Initial score (infinity)
        self.best_score = float("inf")

        for i in range(self.max_iter):
            best_iter_score = float("inf")
            best_iter_expert = None

            # Iterate through all candidates to find the best addition
            for name in expert_names:
                candidate_preds = expert_preds_dict[name]

                # Calculate temporary ensemble average: (Sum + Candidate) / (N + 1)
                temp_sum = current_sum_preds + candidate_preds
                temp_avg = temp_sum / (current_size + 1)

                # Clip probabilities before scoring
                temp_avg_clipped = self._clip_probabilities(temp_avg)

                # Calculate Log Loss
                # Note: sklearn log_loss handles string labels if they match the column order
                # We assume expert_preds are consistent with y_true classes
                try:
                    score = log_loss(y_true, temp_avg_clipped)
                except ValueError as e:
                    # Fallback for potential label mismatches, though unlikely with correct pipeline
                    print(f"Error calculating log_loss for {name}: {e}")
                    score = float("inf")

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            # Calculate improvement
            improvement = self.best_score - best_iter_score

            print(
                f"Iteration {i+1}: Best Candidate = {best_iter_expert}, Score = {best_iter_score:.15f}, Improvement = {improvement:.15f}"
            )

            # Check convergence criteria
            if improvement > self.tol:
                self.best_score = best_iter_score
                self.selected_experts.append(best_iter_expert)
                current_sum_preds += expert_preds_dict[best_iter_expert]
                current_size += 1
            else:
                print(
                    f"Stopping: Improvement {improvement:.15f} < Tolerance {self.tol}"
                )
                break

        # Compute final weights (counts)
        for expert in self.selected_experts:
            self.weights[expert] = self.weights.get(expert, 0) + 1

        print("-" * 30)
        print("Selection Complete.")
        print(f"Selected Experts Counts: {self.weights}")
        print(f"Final Validation Log Loss: {self.best_score:.15f}")
        print("-" * 30)

    def predict(self, expert_preds_dict):
        """
        Generates aggregated predictions using the fitted ensemble.

        Args:
            expert_preds_dict (dict): Dictionary of expert predictions for the test set.

        Returns:
            np.array: Aggregated probabilities (N_samples, N_classes).
        """
        if not self.selected_experts:
            raise ValueError("Selector has not been fitted yet. Call fit() first.")

        # Get dimensions
        first_expert = list(expert_preds_dict.values())[0]
        n_samples, n_classes = first_expert.shape

        final_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        total_weight = 0

        # Weighted aggregation
        for expert, count in self.weights.items():
            if expert in expert_preds_dict:
                final_sum += expert_preds_dict[expert] * count
                total_weight += count
            else:
                raise KeyError(
                    f"Selected expert '{expert}' not found in prediction dictionary."
                )

        if total_weight == 0:
            raise ValueError("Total ensemble weight is zero.")

        final_avg = final_sum / total_weight

        # Apply clipping
        final_avg_clipped = self._clip_probabilities(final_avg)

        return final_avg_clipped
