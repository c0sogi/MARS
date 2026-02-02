import numpy as np
from scipy.optimize import minimize
from library.utils import compute_auc


class Ensemble:
    """
    Ensemble optimization and blending module.
    Handles the calculation of optimal weights for a weighted average ensemble
    to maximize the ROC AUC metric.
    """

    def __init__(self):
        self.weights = None

    def optimize_weights(self, val_preds_list, y_val, initial_weights=None):
        """
        Optimizes the scalar weights for blending multiple model predictions to maximize ROC AUC.

        Args:
            val_preds_list (list of np.ndarray): List of prediction arrays from different models.
                                                 Each array should have shape (num_samples, num_classes).
            y_val (np.ndarray): Ground truth labels with shape (num_samples, num_classes).
            initial_weights (list of float, optional): Initial guess for the weights.
                                                       If None, starts with equal weights.

        Returns:
            np.ndarray: The optimized weights.
        """
        num_models = len(val_preds_list)

        # Validate inputs
        if num_models == 0:
            raise ValueError("val_preds_list cannot be empty.")

        base_shape = val_preds_list[0].shape
        if y_val.shape != base_shape:
            raise ValueError(
                f"Shape mismatch: y_val {y_val.shape} vs preds {base_shape}"
            )

        if initial_weights is None:
            initial_weights = [1.0 / num_models] * num_models

        if len(initial_weights) != num_models:
            raise ValueError("Length of initial_weights must match number of models.")

        # Objective function: Minimize negative AUC
        # We rely on the constraints to keep weights valid, so we don't normalize inside the loop
        def objective(weights):
            blended_preds = np.zeros_like(val_preds_list[0])
            for i, w in enumerate(weights):
                blended_preds += w * val_preds_list[i]

            auc = compute_auc(y_val, blended_preds)
            return -auc

        # Constraints and Bounds
        # Sum of weights = 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        # Weights between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(num_models)]

        print(f"Starting ensemble weight optimization for {num_models} models...")

        # Use SLSQP for constrained optimization
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=1e-6,
            options={"disp": False},
        )

        self.weights = result.x
        best_auc = -result.fun  # Revert sign to get positive AUC

        print("Optimization finished.")
        print(f"Optimized Weights: {self.weights}")
        print(f"Best Validation AUC: {best_auc}")

        return self.weights

    def blend_predictions(self, test_preds_list, weights=None):
        """
        Blends test predictions using the provided or stored weights.

        Args:
            test_preds_list (list of np.ndarray): List of prediction arrays for the test set.
            weights (list or np.ndarray, optional): Weights to apply. If None, uses optimized weights.

        Returns:
            np.ndarray: The weighted average predictions.
        """
        if weights is None:
            if self.weights is None:
                raise ValueError(
                    "No weights available. Run optimize_weights first or provide weights."
                )
            weights = self.weights

        if len(test_preds_list) != len(weights):
            raise ValueError(
                f"Mismatch: {len(test_preds_list)} prediction arrays vs {len(weights)} weights."
            )

        # Compute weighted sum
        blended_preds = np.zeros_like(test_preds_list[0])
        for i, w in enumerate(weights):
            blended_preds += w * test_preds_list[i]

        return blended_preds
