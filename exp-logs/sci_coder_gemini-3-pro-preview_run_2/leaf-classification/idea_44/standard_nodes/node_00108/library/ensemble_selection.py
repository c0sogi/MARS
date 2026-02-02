import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library.config import FLOAT_PRECISION


class GreedyEnsembleSelector:
    """
    Implements Greedy Forward Selection (Caruana et al.) to optimize ensemble weights.

    This algorithm iteratively selects the model that, when added to the current
    ensemble, maximizes the performance (minimizes Log Loss) on the validation set.
    This results in a sparse, integer-weighted linear combination of experts,
    which is robust to overfitting compared to standard regression stacking.
    """

    def __init__(self, n_iterations=100, tol=1e-6):
        """
        Args:
            n_iterations (int): The number of selection steps. Higher values allow
                                for finer-grained weights (e.g., 1/100 = 0.01 steps).
            tol (float): Minimum improvement required to continue (optional).
        """
        self.n_iterations = n_iterations
        self.tol = tol
        self.selected_experts = []
        self.weights = {}
        self.best_score = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using the validation set predictions.

        Args:
            predictions_dict (dict): Dictionary where keys are expert IDs and values
                                     are np.arrays of shape (n_samples, n_classes).
            y_true (np.array): True class labels (n_samples,).

        Returns:
            self
        """
        # Ensure deterministic order of keys for reproducibility
        expert_ids = sorted(list(predictions_dict.keys()))

        # Validate inputs
        if not expert_ids:
            raise ValueError("predictions_dict cannot be empty.")

        n_samples = len(y_true)
        n_classes = predictions_dict[expert_ids[0]].shape[1]

        # Initialize ensemble prediction sum (start with zeros)
        # We maintain the sum to avoid re-averaging the entire history every step
        current_sum = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        self.selected_experts = []
        self.best_score = float("inf")

        print(f"Starting Greedy Forward Selection (Iterations={self.n_iterations})...")

        for i in range(1, self.n_iterations + 1):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each expert to the current ensemble
            for expert_id in expert_ids:
                pred = predictions_dict[expert_id].astype(FLOAT_PRECISION)

                # Calculate temporary average: (current_sum + new_pred) / i
                temp_sum = current_sum + pred
                temp_avg = temp_sum / i

                # Evaluate Log Loss
                # Sklearn log_loss handles the clipping/epsilon internally
                try:
                    score = log_loss(y_true, temp_avg, labels=list(range(n_classes)))
                except ValueError:
                    # Fallback if specific labels are missing in y_true but present in preds
                    score = log_loss(y_true, temp_avg)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = expert_id

            # Update current ensemble with the best expert found in this iteration
            self.selected_experts.append(iteration_best_expert)
            current_sum += predictions_dict[iteration_best_expert].astype(
                FLOAT_PRECISION
            )

            # Check for improvement
            improvement = self.best_score - iteration_best_score
            self.best_score = iteration_best_score

            # Optional: Print progress every 10 steps or first few
            if i <= 5 or i % 10 == 0:
                print(
                    f"Iter {i}/{self.n_iterations}: Added {iteration_best_expert}, "
                    f"Val LogLoss: {self.best_score:.15f}"
                )

            # Early stopping check (optional, usually we run full iterations for weight granularity)
            # if improvement < self.tol and i > 10:
            #     print(f"Stopping early at iteration {i} due to negligible improvement.")
            #     break

        # Calculate final weights
        counts = Counter(self.selected_experts)
        total = len(self.selected_experts)
        self.weights = {k: v / total for k, v in counts.items()}

        print("\nSelection Complete.")
        print(f"Final Validation LogLoss: {self.best_score:.15f}")
        print("Optimized Weights:")
        for expert, weight in self.weights.items():
            print(f"  - {expert}: {weight:.4f}")

        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction for the test set.

        Args:
            predictions_dict (dict): Dictionary where keys are expert IDs and values
                                     are np.arrays of shape (n_samples, n_classes).

        Returns:
            np.array: Weighted probability matrix (n_samples, n_classes).
        """
        if not self.weights:
            raise ValueError("Selector must be fitted before calling predict.")

        # Get shape from first expert
        first_expert = next(iter(predictions_dict.values()))
        n_samples, n_classes = first_expert.shape

        final_pred = np.zeros((n_samples, n_classes), dtype=FLOAT_PRECISION)

        # Accumulate weighted predictions
        for expert_id, weight in self.weights.items():
            if expert_id not in predictions_dict:
                raise KeyError(
                    f"Expert '{expert_id}' selected during fit but missing in predict input."
                )

            pred = predictions_dict[expert_id].astype(FLOAT_PRECISION)
            final_pred += weight * pred

        return final_pred
