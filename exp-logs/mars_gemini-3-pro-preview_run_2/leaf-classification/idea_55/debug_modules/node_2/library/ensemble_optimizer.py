import numpy as np
from sklearn.base import clone
from library.utils import clipped_log_loss
from library.config import FLOAT_PRECISION


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to optimize
    ensemble composition based on validation Log Loss.
    """

    def __init__(self, experts, max_iterations=20, tolerance=1e-6, verbose=True):
        """
        Args:
            experts (list): List of (name, pipeline) tuples.
            max_iterations (int): Maximum number of experts to select (sum of weights).
            tolerance (float): Minimum improvement required to continue selection.
            verbose (bool): Whether to print selection progress.
        """
        self.experts = experts
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        # State
        self.selected_experts_ = []  # List of expert names selected in order
        self.best_loss_ = float("inf")
        self.expert_predictions_ = (
            {}
        )  # Cache for validation predictions: {name: y_pred}
        self.expert_map_ = {name: pipe for name, pipe in experts}

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains all experts and runs the greedy selection process.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.

        Returns:
            self
        """
        # 1. Train all experts and generate validation predictions
        if self.verbose:
            print(
                f"Training {len(self.experts)} experts and generating validation predictions..."
            )

        for name, pipeline in self.experts:
            # Clone to ensure a fresh model
            model = clone(pipeline)

            # Fit on training set
            model.fit(X_train, y_train)

            # Predict on validation set
            # Ensure float64 precision
            preds = model.predict_proba(X_val).astype(FLOAT_PRECISION)
            self.expert_predictions_[name] = preds

        # 2. Greedy Forward Selection
        if self.verbose:
            print("\nStarting Greedy Forward Selection...")

        # Initialize ensemble sum of probabilities
        # Shape: (n_samples, n_classes)
        current_ensemble_sum = np.zeros_like(list(self.expert_predictions_.values())[0])
        current_size = 0

        for i in range(self.max_iterations):
            best_iter_loss = float("inf")
            best_expert_name = None

            # Try adding each expert to the current ensemble
            for name in self.expert_predictions_.keys():
                candidate_preds = self.expert_predictions_[name]

                # Calculate temporary ensemble average
                # New Avg = (Sum_Current + Candidate) / (N + 1)
                temp_sum = current_ensemble_sum + candidate_preds
                temp_avg = temp_sum / (current_size + 1)

                loss = clipped_log_loss(y_val, temp_avg)

                if loss < best_iter_loss:
                    best_iter_loss = loss
                    best_expert_name = name

            # Check for improvement
            improvement = self.best_loss_ - best_iter_loss

            # Update state if improved or if it's the first iteration (initialization)
            if current_size == 0 or improvement > self.tolerance:
                self.best_loss_ = best_iter_loss
                self.selected_experts_.append(best_expert_name)

                # Update the running sum
                current_ensemble_sum += self.expert_predictions_[best_expert_name]
                current_size += 1

                if self.verbose:
                    print(
                        f"Iter {i+1}/{self.max_iterations}: Selected '{best_expert_name}' | "
                        f"Val Loss: {self.best_loss_:.15f} | Improvement: {improvement:.15f}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Iter {i+1}: No significant improvement (Imp: {improvement:.15f} <= Tol: {self.tolerance}). Stopping."
                    )
                break

        return self

    def get_selected_experts(self):
        """
        Returns the selected experts and their counts (weights).

        Returns:
            list: List of tuples (expert_name, count).
        """
        from collections import Counter

        counts = Counter(self.selected_experts_)
        # Sort by most frequent for display clarity, though order doesn't strictly matter for weighted avg
        return counts.most_common()

    def get_best_loss(self):
        """Returns the best validation loss achieved."""
        return self.best_loss_
