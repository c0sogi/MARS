import numpy as np
from collections import Counter
from library.utils import clipped_log_loss


class GreedySelector:
    """
    Implements Greedy Forward Selection (Hill Climbing) for ensemble optimization.
    It iteratively adds the expert that maximizes the reduction in Log Loss (Forward Selection).
    Allows for integer weighting of experts by selecting them multiple times.
    """

    def __init__(self, n_iterations=50, tolerance=1e-6, verbose=True):
        """
        Args:
            n_iterations (int): Maximum number of iterations (experts to select).
            tolerance (float): Minimum improvement in log loss required to continue adding experts.
            verbose (bool): Whether to print progress.
        """
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.selected_experts = []
        self.best_score = float("inf")
        self.weights = {}

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): A dictionary where keys are expert names (str)
                                     and values are prediction arrays (np.ndarray) of shape (N, C).
            y_true (np.ndarray): True class labels of shape (N,).

        Returns:
            self: The fitted selector instance.
        """
        if not predictions_dict:
            raise ValueError("predictions_dict cannot be empty.")

        # Ensure all predictions are float64 for precision
        candidates = {k: v.astype(np.float64) for k, v in predictions_dict.items()}
        expert_names = sorted(
            list(candidates.keys())
        )  # Sort for deterministic order in case of ties

        # Validation of shapes
        n_samples = len(y_true)
        first_expert = expert_names[0]
        n_classes = candidates[first_expert].shape[1]

        if candidates[first_expert].shape[0] != n_samples:
            raise ValueError(
                f"Prediction shape mismatch. Expected {n_samples} samples, got {candidates[first_expert].shape[0]}."
            )

        # Initialize ensemble accumulator (sum of probabilities)
        # We track the sum to avoid re-summing the entire list every iteration
        current_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        current_count = 0

        self.selected_experts = []
        self.best_score = float("inf")

        for i in range(self.n_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each candidate expert to the current ensemble
            for name in expert_names:
                pred = candidates[name]

                # Calculate trial ensemble average
                # Formula: (Current Sum + Candidate Pred) / (Current Count + 1)
                trial_sum = current_sum + pred
                trial_count = current_count + 1
                trial_avg = trial_sum / trial_count

                # Calculate Metric
                score = clipped_log_loss(y_true, trial_avg)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = name

            # Determine improvement
            # For the first iteration, self.best_score is inf, so improvement is inf (always accepts)
            improvement = self.best_score - iteration_best_score

            if self.verbose:
                # Printing full precision as requested
                print(
                    f"Iteration {i+1}: Best Candidate = {iteration_best_expert}, Score = {iteration_best_score}, Improvement = {improvement}"
                )

            # Stopping Criteria 1: No valid candidate found (should not happen normally)
            if iteration_best_expert is None:
                if self.verbose:
                    print("No valid candidate found.")
                break

            # Stopping Criteria 2: Tolerance check (skip for first iteration)
            if current_count > 0 and improvement < self.tolerance:
                if self.verbose:
                    print(
                        f"Improvement {improvement} is below tolerance {self.tolerance}. Stopping."
                    )
                break

            # Update Ensemble State
            self.best_score = iteration_best_score
            self.selected_experts.append(iteration_best_expert)
            current_sum += candidates[iteration_best_expert]
            current_count += 1

        # Calculate final integer weights
        self.weights = dict(Counter(self.selected_experts))

        if self.verbose:
            print("Selection Complete.")
            print(f"Selected Experts: {self.selected_experts}")
            print(f"Weights: {self.weights}")
            print(f"Final Validation Score: {self.best_score}")

        return self

    def get_best_weights(self):
        """
        Returns the dictionary of selected experts and their integer weights.

        Returns:
            dict: {expert_name: count}
        """
        return self.weights
