import numpy as np
from scipy.optimize import minimize
from library.config import Config
from library.utils import compute_log_loss, clip_probabilities


class EnsembleOptimizer:
    """
    Optimizes ensemble weights using scipy.optimize.minimize to minimize Log Loss
    on Out-Of-Fold (OOF) predictions.
    """

    def __init__(self):
        self.weights = None
        self.model_names = None

    def optimize_weights(self, oof_preds_dict, y_true):
        """
        Finds the optimal weights for blending model predictions.

        Args:
            oof_preds_dict (dict): Dictionary where keys are model names and values
                                   are np.arrays of shape (n_samples, n_classes)
                                   containing OOF probabilities.
            y_true (np.array): Ground truth labels (class indices).

        Returns:
            dict: Optimized weights mapping {model_name: weight}.
        """
        # Sort keys to ensure consistent order of processing
        self.model_names = sorted(list(oof_preds_dict.keys()))

        # Prepare list of prediction arrays corresponding to the sorted keys
        predictions_list = [oof_preds_dict[name] for name in self.model_names]

        # Validation
        if not predictions_list:
            raise ValueError("No predictions provided for optimization.")

        n_models = len(predictions_list)
        n_samples = len(y_true)

        # Verify shapes match y_true
        for name, preds in oof_preds_dict.items():
            if len(preds) != n_samples:
                raise ValueError(
                    f"Length mismatch: y_true has {n_samples}, {name} has {len(preds)}"
                )

        # Define the Objective Function for minimization
        def loss_func(weights):
            # weights is a numpy array of shape (n_models,) provided by the optimizer

            # Compute weighted average of predictions
            final_pred = np.zeros_like(predictions_list[0])
            for i, w in enumerate(weights):
                final_pred += w * predictions_list[i]

            # compute_log_loss handles normalization (dividing by row sum)
            # and clipping internally, so we pass the raw weighted sum.
            return compute_log_loss(y_true, final_pred)

        # Constraints and Bounds
        # Constraint: Sum of weights must equal 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: Each weight must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Initial Guess: Equal weights for all models
        initial_weights = np.ones(n_models) / n_models

        print(f"Optimizing ensemble weights for models: {self.model_names}...")

        # Run Optimization
        result = minimize(
            loss_func,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},  # We print results manually
        )

        # Store results
        optimized_weights_array = result.x
        self.weights = {
            name: w for name, w in zip(self.model_names, optimized_weights_array)
        }

        print("Optimization Complete.")
        print(f"Success: {result.success}")
        print(f"Optimized Log Loss: {result.fun}")
        print("Weights:")
        for name, w in self.weights.items():
            print(f"  {name}: {w}")

        return self.weights

    def blend_predictions(self, preds_dict, weights=None):
        """
        Blends predictions using the specified or stored weights.

        Args:
            preds_dict (dict): Dictionary of predictions {model_name: probs}.
            weights (dict, optional): Dictionary of weights. Uses stored weights if None.

        Returns:
            np.array: Blended probability matrix.
        """
        if weights is None:
            if self.weights is None:
                raise ValueError("Weights not provided and optimization not run.")
            weights = self.weights

        # Validate that we have predictions for all weights
        missing_models = [name for name in weights.keys() if name not in preds_dict]
        if missing_models:
            raise KeyError(
                f"Predictions missing for models in weights: {missing_models}"
            )

        # Initialize accumulator
        # We take the shape from the first model found in the weights
        first_model = list(weights.keys())[0]
        final_pred = np.zeros_like(preds_dict[first_model])

        # Compute Weighted Sum
        for name, weight in weights.items():
            final_pred += weight * preds_dict[name]

        # Normalize rows to sum to 1
        # Although optimization constraints ensure weights sum to 1,
        # floating point arithmetic or manual weight overrides might cause drift.
        row_sums = final_pred.sum(axis=1, keepdims=True)

        # Avoid division by zero
        final_pred = np.divide(
            final_pred,
            row_sums,
            out=np.zeros_like(final_pred),
            where=row_sums != 0,
        )

        # Clip probabilities to avoid log(0) issues and comply with metric spec
        return clip_probabilities(final_pred)
