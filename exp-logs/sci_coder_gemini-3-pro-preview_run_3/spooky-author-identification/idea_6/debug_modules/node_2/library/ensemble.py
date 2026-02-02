import os
import numpy as np
from scipy.optimize import minimize
from library.config import Config
from library.utils import calculate_log_loss, format_submission


class EnsembleOptimizer:
    """
    Optimizes the weights of a weighted average ensemble to minimize Log Loss.
    Uses SciPy's SLSQP optimization method with constraints (sum of weights = 1)
    and bounds (0 <= weight <= 1).
    """

    def __init__(self, model_names):
        """
        Args:
            model_names (list of str): List of names/identifiers for the models being ensembled.
                                       Used for logging and tracking weights.
        """
        self.model_names = model_names
        self.weights = None
        self.num_models = len(model_names)

    def optimize(self, oof_preds_list, y_true):
        """
        Finds the optimal weights for blending the provided OOF predictions.

        Args:
            oof_preds_list (list of np.ndarray): List of OOF prediction arrays.
                                                 Each array should have shape (n_samples, n_classes).
            y_true (np.ndarray): True labels for the OOF data. Shape (n_samples,).

        Returns:
            np.ndarray: The optimal weights.
            float: The optimized log loss score.
        """
        # validation
        if len(oof_preds_list) != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} prediction arrays, got {len(oof_preds_list)}"
            )

        # Define the objective function to minimize
        def loss_func(weights):
            # Calculate weighted average
            final_pred = np.zeros_like(oof_preds_list[0])
            for i, w in enumerate(weights):
                final_pred += w * oof_preds_list[i]

            # Calculate Log Loss
            # Note: calculate_log_loss handles clipping internally via sklearn or utility logic
            return calculate_log_loss(y_true, final_pred)

        # Constraints: Sum of weights must be 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: Weights must be between 0 and 1
        bounds = [(0.0, 1.0)] * self.num_models

        # Initial Guess: Equal weights
        initial_weights = [1.0 / self.num_models] * self.num_models

        print(
            f"Starting ensemble optimization for {self.num_models} models: {self.model_names}..."
        )

        # Run Optimization
        result = minimize(
            loss_func,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "ftol": 1e-9},
        )

        self.weights = result.x
        best_score = result.fun

        # Log results
        print("-" * 40)
        print("Optimization Complete.")
        print(f"Best OOF Log Loss: {best_score:.10f}")
        print("Optimal Weights:")
        for name, weight in zip(self.model_names, self.weights):
            print(f"  {name}: {weight:.6f}")
        print("-" * 40)

        return self.weights, best_score

    def predict(self, test_preds_list):
        """
        Applies the optimized weights to a list of test predictions.

        Args:
            test_preds_list (list of np.ndarray): List of test prediction arrays.

        Returns:
            np.ndarray: The weighted average predictions.
        """
        if self.weights is None:
            raise RuntimeError("Optimizer has not been run. Call optimize() first.")

        if len(test_preds_list) != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} prediction arrays, got {len(test_preds_list)}"
            )

        final_pred = np.zeros_like(test_preds_list[0])
        for i, w in enumerate(self.weights):
            final_pred += w * test_preds_list[i]

        return final_pred

    def generate_submission(
        self, test_preds, test_ids, output_filename="submission.csv"
    ):
        """
        Formats and saves the submission file.

        Args:
            test_preds (np.ndarray): The final blended predictions.
            test_ids (np.ndarray): The IDs corresponding to the predictions.
            output_filename (str): Name of the output file.
        """
        output_path = os.path.join(Config.SUBMISSION_DIR, output_filename)

        # Get column names from Config
        columns = [Config.ID2LABEL[i] for i in range(Config.NUM_CLASSES)]

        # Use the utility function to format and save
        format_submission(test_ids, test_preds, columns, output_path)

        print(f"Submission saved successfully to: {output_path}")
