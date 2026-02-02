import numpy as np
from scipy.optimize import minimize
from library.utils import set_seed, clip_log_loss


class WeightOptimizer:
    """
    Optimizes ensemble weights to minimize log loss using SLSQP.
    Ensures weights are non-negative and sum to one.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the optimizer.

        Args:
            random_state (int): Seed for reproducibility.
        """
        set_seed(random_state)
        self.weights = None
        self.random_state = random_state

    def fit(self, preds_list, y_true, classes=None):
        """
        Finds the optimal weights for the ensemble by minimizing log loss on provided predictions.

        Args:
            preds_list (list of np.ndarray): List of prediction arrays from different models (e.g., OOF preds).
                                             Each array should have shape (n_samples, n_classes).
            y_true (np.ndarray): True class labels (indices or strings).
            classes (np.ndarray, optional): List of all unique class labels.

        Returns:
            self: The fitted optimizer instance.
        """
        n_models = len(preds_list)
        if n_models == 0:
            raise ValueError("preds_list cannot be empty")

        # Initial guess: equal weights for all models
        initial_weights = np.ones(n_models) / n_models

        # Constraints: sum of weights must equal 1
        # eq: fun(x) == 0  => sum(w) - 1 = 0
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: each weight must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Objective function to minimize
        def loss_func(weights):
            # Calculate weighted average of predictions
            # We initialize with zeros
            final_pred = np.zeros_like(preds_list[0])
            for i, w in enumerate(weights):
                final_pred += w * preds_list[i]

            # Calculate log loss using the competition metric utility
            # This utility handles row-wise normalization and clipping
            return clip_log_loss(y_true, final_pred, classes=classes)

        print(f"Optimizing ensemble weights for {n_models} models...")

        # Perform optimization
        result = minimize(
            loss_func,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )

        self.weights = result.x

        # Normalize weights to ensure they strictly sum to 1 (handling potential float precision issues)
        self.weights = self.weights / np.sum(self.weights)

        final_score = result.fun
        print(f"Optimization Success: {result.success}")
        print(f"Optimized Weights: {self.weights}")
        print(f"Ensemble OOF Log Loss: {final_score}")

        return self

    def predict(self, preds_list):
        """
        Generates ensemble predictions using the fitted weights.

        Args:
            preds_list (list of np.ndarray): List of prediction arrays for test data.
                                             Must correspond to the models used in fit().

        Returns:
            np.ndarray: The weighted average predictions.
        """
        if self.weights is None:
            raise ValueError("Optimizer not fitted yet. Call fit() first.")

        if len(preds_list) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} prediction arrays, got {len(preds_list)}"
            )

        # Compute weighted average
        final_pred = np.zeros_like(preds_list[0])
        for i, w in enumerate(self.weights):
            final_pred += w * preds_list[i]

        return final_pred
