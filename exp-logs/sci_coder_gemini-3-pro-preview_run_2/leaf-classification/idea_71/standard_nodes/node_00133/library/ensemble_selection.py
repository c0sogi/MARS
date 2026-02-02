import numpy as np
from collections import Counter
from library.utils import clipped_log_loss
from library.config import MAX_SELECTION_STEPS, SELECTION_TOLERANCE


class GreedySelector:
    """
    Implements Greedy Forward Selection for ensemble optimization.
    Iteratively adds experts to the ensemble (with replacement) to minimize
    validation log loss.
    """

    def __init__(self, max_steps=MAX_SELECTION_STEPS, tolerance=SELECTION_TOLERANCE):
        """
        Args:
            max_steps (int): Maximum number of selection iterations.
            tolerance (float): Minimum improvement in log loss required to continue.
        """
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.selected_experts = []  # List of expert names in order of selection
        self.best_loss = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Selects experts based on validation performance.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to prediction matrices (n_samples, n_classes).
            y_true (array-like): True validation labels.
        """
        # Ensure inputs are float64 for precision
        for name in predictions_dict:
            predictions_dict[name] = predictions_dict[name].astype(np.float64)

        available_experts = list(predictions_dict.keys())

        # Get shape from the first available prediction matrix
        if not available_experts:
            raise ValueError("predictions_dict cannot be empty.")

        sample_shape = predictions_dict[available_experts[0]].shape

        # Running sum of probabilities for the current ensemble
        # Initialize with zeros
        current_ensemble_sum = np.zeros(sample_shape, dtype=np.float64)

        print(
            f"Starting Greedy Forward Selection (Max Steps: {self.max_steps}, Tolerance: {self.tolerance})"
        )

        for step in range(1, self.max_steps + 1):
            best_step_expert = None
            best_step_loss = float("inf")

            # Try adding each available expert to the current ensemble
            for expert_name in available_experts:
                candidate_probs = predictions_dict[expert_name]

                # Calculate temporary ensemble average
                # New Avg = (Sum of previous + Candidate) / Current Step Count
                temp_ensemble_probs = (current_ensemble_sum + candidate_probs) / step

                # Calculate Loss
                # clipped_log_loss handles row normalization and clipping internally
                loss = clipped_log_loss(y_true, temp_ensemble_probs)

                if loss < best_step_loss:
                    best_step_loss = loss
                    best_step_expert = expert_name

            # Check for improvement
            improvement = self.best_loss - best_step_loss

            # Logic for stopping:
            # If it's the first step, we always accept the best single model (improvement is inf -> valid).
            # Afterwards, we check if the improvement meets the tolerance.
            if step > 1 and improvement < self.tolerance:
                print(
                    f"Step {step}: Improvement {improvement:.15f} < Tolerance {self.tolerance}. Stopping."
                )
                break

            # Update state
            self.best_loss = best_step_loss
            self.selected_experts.append(best_step_expert)
            current_ensemble_sum += predictions_dict[best_step_expert]

            print(
                f"Step {step}: Added {best_step_expert}, Validation Loss: {self.best_loss:.15f}"
            )

        print("Selection Complete.")
        print(f"Selected {len(self.selected_experts)} experts.")

        # Print weights for information
        counts = Counter(self.selected_experts)
        print("Ensemble Weights:")
        for name, count in counts.items():
            print(f"  {name}: {count}")

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary mapping expert names to prediction matrices.

        Returns:
            np.ndarray: The ensemble prediction matrix.
        """
        if not self.selected_experts:
            raise ValueError("No experts selected. Call fit() first.")

        # Get shape from an arbitrary expert in the dict
        sample_shape = next(iter(predictions_dict.values())).shape
        ensemble_sum = np.zeros(sample_shape, dtype=np.float64)

        # Accumulate predictions based on selection frequency (weights)
        for expert_name in self.selected_experts:
            if expert_name not in predictions_dict:
                raise KeyError(
                    f"Selected expert '{expert_name}' not found in predictions dictionary."
                )

            ensemble_sum += predictions_dict[expert_name].astype(np.float64)

        # Average
        ensemble_avg = ensemble_sum / len(self.selected_experts)

        return ensemble_avg
