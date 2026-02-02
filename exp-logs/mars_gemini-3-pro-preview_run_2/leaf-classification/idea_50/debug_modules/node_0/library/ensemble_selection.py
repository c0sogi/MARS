import numpy as np
from collections import Counter
from library.utils import clipped_log_loss


class GreedyEnsembleSelector:
    """
    Implements the Greedy Forward Selection algorithm with replacement to optimize
    ensemble weights by iteratively adding experts that maximize validation metric improvement.
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Args:
            max_iterations (int): Maximum number of iterations (experts to add).
            tolerance (float): Minimum improvement in log loss required to continue adding experts.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.selected_experts = []
        self.weights = {}
        self.best_score = float("inf")

    def fit(self, predictions_dict: dict, y_true: np.ndarray):
        """
        Fits the ensemble weights using greedy forward selection on validation data.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values are
                                     prediction probability matrices of shape (n_samples, n_classes).
            y_true (np.ndarray): True class labels of shape (n_samples,).

        Returns:
            tuple: (weights_dict, best_score) where weights_dict maps expert names to integer counts.
        """
        expert_names = list(predictions_dict.keys())
        if not expert_names:
            raise ValueError("predictions_dict cannot be empty.")

        # Initialize ensemble state
        # We maintain the sum of predictions to avoid re-summing the whole history every iteration
        # Get shape from the first expert's predictions
        sample_pred = predictions_dict[expert_names[0]]
        current_ensemble_sum = np.zeros_like(sample_pred)
        current_total_weight = 0

        self.selected_experts = []
        self.best_score = float("inf")

        print(
            f"Starting Greedy Forward Selection with {len(expert_names)} candidates..."
        )

        for i in range(self.max_iterations):
            best_iteration_score = float("inf")
            best_candidate = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                candidate_pred = predictions_dict[name]

                # Calculate trial ensemble prediction
                # Formula: (Current Sum + Candidate) / (Current Weight + 1)
                # This represents the average if we add this candidate
                trial_pred = (current_ensemble_sum + candidate_pred) / (
                    current_total_weight + 1
                )

                # Calculate metric
                score = clipped_log_loss(y_true, trial_pred)

                if score < best_iteration_score:
                    best_iteration_score = score
                    best_candidate = name

            # Check for improvement
            # On the first iteration, we always accept the best single model (improvement is effectively infinite)
            if i == 0:
                improvement = float("inf")
            else:
                improvement = self.best_score - best_iteration_score

            if improvement > self.tolerance:
                self.best_score = best_iteration_score
                self.selected_experts.append(best_candidate)

                # Update the running sum and weight
                current_ensemble_sum += predictions_dict[best_candidate]
                current_total_weight += 1

                print(
                    f"Iteration {i+1}: Added expert '{best_candidate}' with score {self.best_score}"
                )
            else:
                print(
                    f"Iteration {i+1}: Best improvement {improvement} <= Tolerance {self.tolerance}. Stopping."
                )
                break

        # Calculate final integer weights (counts of each expert)
        self.weights = dict(Counter(self.selected_experts))

        print("Selection Complete.")
        print(f"Final Weights: {self.weights}")
        print(f"Best Validation Score: {self.best_score}")

        return self.weights, self.best_score

    def predict(self, predictions_dict: dict):
        """
        Generates ensemble predictions using the fitted weights.

        Args:
            predictions_dict (dict): Dictionary of expert predictions (e.g., on test set).

        Returns:
            np.ndarray: Weighted average prediction matrix.
        """
        if not self.weights:
            raise RuntimeError(
                "EnsembleSelector has not been fitted or no experts were selected."
            )

        final_pred = None
        total_weight = 0

        for name, weight in self.weights.items():
            if name not in predictions_dict:
                raise KeyError(
                    f"Expert '{name}' found in weights but missing from input predictions."
                )

            pred = predictions_dict[name]

            if final_pred is None:
                final_pred = np.zeros_like(pred)

            final_pred += weight * pred
            total_weight += weight

        if total_weight == 0:
            raise RuntimeError("Total weight is zero.")

        return final_pred / total_weight
