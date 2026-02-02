import numpy as np
from collections import Counter
from library.utils import clipped_log_loss
from library.config import SELECTION_ITERATIONS, SELECTION_TOLERANCE, FLOAT_PRECISION


class GreedySelector:
    """
    Implements Greedy Forward Selection (Hill Climbing) for ensemble optimization.

    This class iteratively builds an ensemble by adding the expert that maximizes
    the improvement in the evaluation metric (Log Loss) at each step. It supports
    selection with replacement, effectively assigning integer weights to experts.
    """

    def __init__(
        self, n_iterations=SELECTION_ITERATIONS, tolerance=SELECTION_TOLERANCE
    ):
        """
        Args:
            n_iterations (int): Maximum number of experts to add to the ensemble.
            tolerance (float): Minimum improvement in Log Loss required to continue adding experts.
        """
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.selected_experts = []  # List of expert names in order of selection
        self.weights = {}  # Dictionary of {expert_name: count}
        self.best_score = float("inf")
        self.trajectory = []  # Stores (iteration, score) for analysis

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values
                                     are numpy arrays of shape (n_samples, n_classes)
                                     containing predicted probabilities.
            y_true (np.array): Ground truth labels (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        # Ensure inputs are in correct precision
        expert_names = list(predictions_dict.keys())

        # Pre-cast all predictions to float64 to avoid repeated casting during the loop
        # and ensure numerical stability.
        cached_preds = {
            k: v.astype(FLOAT_PRECISION) for k, v in predictions_dict.items()
        }

        print(
            f"Starting Greedy Forward Selection with {len(expert_names)} candidates..."
        )
        print(f"Max Iterations: {self.n_iterations}, Tolerance: {self.tolerance}")

        # =====================================================================
        # Step 1: Initialization - Find the single best expert
        # =====================================================================
        initial_best_loss = float("inf")
        initial_best_expert = None

        for name, preds in cached_preds.items():
            loss = clipped_log_loss(y_true, preds)
            if loss < initial_best_loss:
                initial_best_loss = loss
                initial_best_expert = name

        if initial_best_expert is None:
            raise ValueError("No experts provided or evaluation failed.")

        # Initialize ensemble state
        self.selected_experts = [initial_best_expert]
        self.best_score = initial_best_loss

        # We maintain the sum of probabilities of the current ensemble to avoid re-summing
        # O(N*K) complexity where N is ensemble size.
        # current_sum_probs shape: (n_samples, n_classes)
        current_sum_probs = cached_preds[initial_best_expert].copy()
        current_size = 1.0

        self.trajectory.append((0, self.best_score))
        print(f"Iter 0: Selected {initial_best_expert} | Score: {self.best_score:.15f}")

        # =====================================================================
        # Step 2: Iterative Selection
        # =====================================================================
        for i in range(1, self.n_iterations + 1):
            iteration_best_loss = float("inf")
            iteration_best_expert = None

            # Try adding each expert to the current ensemble
            for name, preds in cached_preds.items():
                # Calculate temporary ensemble prediction
                # New Average = (Sum + New_Preds) / (Size + 1)
                temp_sum = current_sum_probs + preds
                temp_preds = temp_sum / (current_size + 1.0)

                loss = clipped_log_loss(y_true, temp_preds)

                if loss < iteration_best_loss:
                    iteration_best_loss = loss
                    iteration_best_expert = name

            # Check for improvement
            improvement = self.best_score - iteration_best_loss

            if improvement > self.tolerance:
                # Update state
                self.best_score = iteration_best_loss
                self.selected_experts.append(iteration_best_expert)
                current_sum_probs += cached_preds[iteration_best_expert]
                current_size += 1.0

                self.trajectory.append((i, self.best_score))
                print(
                    f"Iter {i}: Added {iteration_best_expert} | Score: {self.best_score:.15f} | Improv: {improvement:.15f}"
                )
            else:
                print(
                    f"Iter {i}: Improvement {improvement:.15f} < Tolerance {self.tolerance}. Stopping."
                )
                break

        # =====================================================================
        # Step 3: Finalize Weights
        # =====================================================================
        self.weights = dict(Counter(self.selected_experts))

        print("\nSelection Complete.")
        print(f"Final Ensemble Size: {len(self.selected_experts)}")
        print(f"Final Validation Log Loss: {self.best_score:.15f}")
        print("Selected Experts and Weights:")
        for name, weight in self.weights.items():
            print(f"  - {name}: {weight}")

        return self

    def get_best_weights(self):
        """
        Returns the dictionary of selected experts and their integer weights.
        """
        return self.weights

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary of expert predictions (e.g., on Test set).

        Returns:
            np.array: Weighted average probabilities (n_samples, n_classes).
        """
        if not self.weights:
            raise ValueError("Selector has not been fitted yet.")

        # Initialize sum
        first_key = next(iter(predictions_dict))
        n_samples, n_classes = predictions_dict[first_key].shape
        weighted_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)
        total_weight = 0.0

        for name, weight in self.weights.items():
            if name not in predictions_dict:
                raise KeyError(
                    f"Selected expert '{name}' not found in provided predictions."
                )

            preds = predictions_dict[name].astype(FLOAT_PRECISION)
            weighted_sum += preds * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight is zero.")

        final_preds = weighted_sum / total_weight
        return final_preds
