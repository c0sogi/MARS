import numpy as np
from collections import Counter
from library.utils import calculate_log_loss, set_seed
from library.config import RANDOM_SEED


class GreedyEnsemble:
    """
    Implements Greedy Forward Selection for ensemble optimization.
    Selects experts from a library to minimize log loss on a validation set.
    """

    def __init__(self, max_iterations=100, tol=1e-6):
        """
        Args:
            max_iterations (int): Maximum number of experts to add.
            tol (float): Minimum improvement required to continue.
        """
        self.max_iterations = max_iterations
        self.tol = tol
        self.weights = {}  # Maps expert_name -> integer weight
        self.selected_experts = []  # List of selected expert names in order
        self.best_score = float("inf")
        set_seed(RANDOM_SEED)

    def fit(self, library_preds, y_true, verbose=True):
        """
        Fits the ensemble weights using Greedy Forward Selection.

        Args:
            library_preds (dict): Dictionary {expert_name: prediction_matrix}.
                                  prediction_matrix shape: (n_samples, n_classes).
            y_true (np.ndarray): True labels (n_samples,).
            verbose (bool): Whether to print progress.
        """
        expert_names = list(library_preds.keys())
        if not expert_names:
            raise ValueError("Library of predictions is empty.")

        # Ensure predictions are float64 for precision
        for name in expert_names:
            library_preds[name] = library_preds[name].astype(np.float64)

        # Initialize ensemble state
        # We start with 0 predictions. The first step will pick the single best model.
        # current_sum_preds will accumulate weighted sum of predictions
        # We use the shape of the first expert's predictions to initialize zeros
        sample_shape = list(library_preds.values())[0].shape
        current_sum_preds = np.zeros(sample_shape, dtype=np.float64)
        current_total_weight = 0

        self.selected_experts = []
        self.weights = {name: 0 for name in expert_names}
        self.best_score = float("inf")

        if verbose:
            print(
                f"Starting Greedy Forward Selection with {len(expert_names)} experts..."
            )

        for i in range(self.max_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                preds = library_preds[name]

                # Calculate candidate ensemble predictions
                # Formula: (current_sum + new_pred) / (current_weight + 1)
                candidate_preds = (current_sum_preds + preds) / (
                    current_total_weight + 1
                )

                # Calculate score
                # calculate_log_loss handles row normalization and clipping internally
                # Cite debug_lesson_7: Explicitly pass labels to handle sparse validation sets
                labels = np.arange(candidate_preds.shape[1])
                score = calculate_log_loss(y_true, candidate_preds, labels=labels)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = name

            # Check for improvement
            # For the first iteration, we always accept the best single model
            # For subsequent iterations, we check tolerance
            improvement = self.best_score - iteration_best_score

            if verbose:
                print(
                    f"Iteration {i+1}: Best candidate '{iteration_best_expert}' with score {iteration_best_score}"
                )

            if iteration_best_expert is None:
                if verbose:
                    print("No valid candidate found.")
                break

            # If it's the first iteration, or improvement > tol
            if (i == 0) or (improvement > self.tol):
                self.best_score = iteration_best_score
                self.selected_experts.append(iteration_best_expert)
                self.weights[iteration_best_expert] += 1

                # Update current ensemble state
                current_sum_preds += library_preds[iteration_best_expert]
                current_total_weight += 1

                if verbose:
                    print(f"  Selected: {iteration_best_expert}")
                    print(f"  New Ensemble Score: {self.best_score}")
            else:
                if verbose:
                    print(
                        f"  Improvement {improvement} <= tolerance {self.tol}. Stopping."
                    )
                break

        # Filter weights to only include selected experts
        self.weights = {k: v for k, v in self.weights.items() if v > 0}

        if verbose:
            print("Selection Complete.")
            print(f"Final Score: {self.best_score}")
            print(f"Selected Experts: {dict(Counter(self.selected_experts))}")

    def predict(self, library_preds):
        """
        Generates aggregated predictions for a new set of data (e.g., test set).

        Args:
            library_preds (dict): Dictionary {expert_name: prediction_matrix}.

        Returns:
            np.ndarray: Aggregated probability matrix.
        """
        if not self.weights:
            raise ValueError("Ensemble is not fitted or no experts were selected.")

        # Initialize sum
        first_expert = next(iter(self.weights.keys()))
        if first_expert not in library_preds:
            raise KeyError(
                f"Expert '{first_expert}' selected during fit not found in prediction library."
            )

        sum_preds = np.zeros_like(library_preds[first_expert], dtype=np.float64)
        total_weight = 0

        for name, weight in self.weights.items():
            if name not in library_preds:
                raise KeyError(
                    f"Expert '{name}' selected during fit not found in prediction library."
                )

            preds = library_preds[name].astype(np.float64)
            sum_preds += preds * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total ensemble weight is zero.")

        # Return weighted average
        # Note: We do not clip here; clipping is handled by the scoring function or final submission preparation
        # However, the metric function provided in utils handles normalization.
        # Here we return the raw weighted average which sums to 1 (since inputs sum to 1).
        return sum_preds / total_weight
