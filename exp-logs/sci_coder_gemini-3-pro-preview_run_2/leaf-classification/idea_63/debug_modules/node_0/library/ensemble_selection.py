import numpy as np
from collections import Counter
from library.utils import clipped_log_loss


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement for ensemble optimization.
    This selector iteratively adds models to the ensemble that maximize the metric (minimize log loss).
    """

    def __init__(self, max_iter=100, tolerance=1e-6):
        """
        Args:
            max_iter (int): Maximum number of iterations (experts to add).
            tolerance (float): Minimum improvement in log loss required to continue adding experts.
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.selected_experts = []  # List of expert names in order of selection
        self.weights = {}  # Dictionary mapping expert_name -> count (weight)
        self.best_loss = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble selection by iteratively adding experts that minimize log loss.

        Args:
            predictions_dict (dict): Dictionary {expert_name: prediction_matrix (N_samples, N_classes)}.
            y_true (np.array): Ground truth labels (N_samples,).
        """
        expert_names = list(predictions_dict.keys())
        if not expert_names:
            raise ValueError("predictions_dict cannot be empty.")

        print(
            f"Starting Greedy Forward Selection (Max Iter: {self.max_iter}, Tol: {self.tolerance})"
        )

        # ---------------------------------------------------------------------
        # 1. Initialization: Find the single best model
        # ---------------------------------------------------------------------
        best_single_loss = float("inf")
        best_single_expert = None

        for name in expert_names:
            # Compute loss for single model
            loss = clipped_log_loss(y_true, predictions_dict[name])
            if loss < best_single_loss:
                best_single_loss = loss
                best_single_expert = name

        # Initialize ensemble with the best single model
        self.selected_experts = [best_single_expert]
        self.best_loss = best_single_loss

        # Current ensemble prediction (starts as the best single model's prediction)
        # We use float64 for accumulation to minimize numerical error
        current_ensemble_pred = predictions_dict[best_single_expert].astype(np.float64)

        print(f"Iter 0: Selected '{best_single_expert}' | Loss: {self.best_loss:.10f}")

        # ---------------------------------------------------------------------
        # 2. Iterative Selection
        # ---------------------------------------------------------------------
        # We start loop from 1 because we already have 1 model
        for i in range(1, self.max_iter + 1):
            best_iter_loss = float("inf")
            best_iter_expert = None

            # Current ensemble has size 'i'. We are looking for the (i+1)-th member.
            # Formula for new average: P_new = (Sum_current + P_candidate) / (i + 1)
            # Optimization: We maintain the current average 'current_ensemble_pred'.
            # Sum_current = current_ensemble_pred * i

            current_sum = current_ensemble_pred * i

            for name in expert_names:
                candidate_pred = predictions_dict[name]

                # Calculate potential new ensemble prediction
                # Note: We divide by (i + 1) to get the mean
                new_pred = (current_sum + candidate_pred) / (i + 1)

                loss = clipped_log_loss(y_true, new_pred)

                if loss < best_iter_loss:
                    best_iter_loss = loss
                    best_iter_expert = name

            # Check for improvement
            improvement = self.best_loss - best_iter_loss

            if improvement > self.tolerance:
                self.best_loss = best_iter_loss
                self.selected_experts.append(best_iter_expert)

                # Update current ensemble prediction to the new best state
                current_ensemble_pred = (
                    current_sum + predictions_dict[best_iter_expert]
                ) / (i + 1)

                print(
                    f"Iter {i}: Added '{best_iter_expert}' | Loss: {self.best_loss:.10f} | Improv: {improvement:.10f}"
                )
            else:
                print(
                    f"Iter {i}: Best improvement {improvement:.10f} < tolerance {self.tolerance}. Stopping."
                )
                break

        # ---------------------------------------------------------------------
        # 3. Finalize Weights
        # ---------------------------------------------------------------------
        self.weights = dict(Counter(self.selected_experts))

        print("\nSelection Complete.")
        print(f"Final Ensemble Size: {len(self.selected_experts)}")
        print(f"Final Validation Loss: {self.best_loss:.10f}")
        print("Selected Experts and Weights:")
        for name, count in sorted(
            self.weights.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  {name}: {count}")

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary {expert_name: prediction_matrix (N, C)}.

        Returns:
            np.array: Ensemble probabilities (N, C).
        """
        if not self.selected_experts:
            raise ValueError("Selector has not been fitted yet.")

        # Get shape from the first expert
        first_expert_name = list(predictions_dict.keys())[0]
        n_samples, n_classes = predictions_dict[first_expert_name].shape

        # Initialize accumulator
        ensemble_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        total_weight = 0

        for name, count in self.weights.items():
            if name not in predictions_dict:
                raise KeyError(
                    f"Expert '{name}' selected during fit not found in prediction dictionary."
                )

            # Weighted sum
            ensemble_sum += predictions_dict[name] * count
            total_weight += count

        # Compute mean
        if total_weight == 0:
            raise ValueError("Total weight is zero.")

        return ensemble_sum / total_weight

    def get_selected_experts(self):
        """
        Returns the dictionary of selected experts and their weights.
        """
        return self.weights
