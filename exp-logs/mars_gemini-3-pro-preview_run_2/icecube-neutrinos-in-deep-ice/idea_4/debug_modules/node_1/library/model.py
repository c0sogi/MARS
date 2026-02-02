import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import (
    LGBM_PARAMS,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    MODEL_X_PATH,
    MODEL_Y_PATH,
    MODEL_Z_PATH,
    SEED,
)
from library.utils import setup_logger, cartesian_to_spherical


class GradientBoostingVectorRegressor:
    """
    A regressor that predicts the 3D direction vector (x, y, z) of a neutrino event
    using three independent Gradient Boosting Machines (LightGBM).
    The predictions are then converted to spherical coordinates (azimuth, zenith).
    """

    def __init__(self):
        self.logger = setup_logger("GBVR")
        self.models = {}
        self.model_paths = {"x": MODEL_X_PATH, "y": MODEL_Y_PATH, "z": MODEL_Z_PATH}
        # Ensure the directory for models exists
        for path in self.model_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the three vector component models (x, y, z).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets (must contain 'target_x', 'target_y', 'target_z').
            X_val (pd.DataFrame, optional): Validation features for early stopping.
            y_val (pd.DataFrame, optional): Validation targets.

        Returns:
            dict: Best validation MSE scores for each component.
        """
        self.logger.info("Starting training of GradientBoostingVectorRegressor...")
        metrics = {}

        for axis in ["x", "y", "z"]:
            self.logger.info(f"Training model for target component: {axis}")
            target_col = f"target_{axis}"

            # Initialize LightGBM Regressor
            # We pass the global parameters from config
            model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGBM_PARAMS)

            # Configure Early Stopping and Evaluation
            callbacks = []
            eval_set = None

            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[target_col])]
                # Add early stopping callback
                # verbose=False suppresses the per-iteration log, we print the final result manually
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    )
                )
                # Add log_evaluation callback with period=0 to suppress default logging
                callbacks.append(lgb.log_evaluation(period=0))

            # Train the model
            model.fit(
                X_train,
                y_train[target_col],
                eval_set=eval_set,
                eval_metric="mse",
                callbacks=callbacks,
            )

            # Store the trained model instance
            self.models[axis] = model

            # Save the model artifact
            # We use the booster's save_model method for a lightweight text representation
            model.booster_.save_model(self.model_paths[axis])

            # Retrieve and log the best validation score
            if eval_set:
                # best_score_ structure: {'valid_0': {'l2': 0.12345...}}
                # Note: 'l2' is the default metric key for regression/mse in LightGBM
                val_scores = model.best_score_["valid_0"]
                # Get the score (l2 or mse)
                score = val_scores.get(
                    "l2", val_scores.get("mse", list(val_scores.values())[0])
                )

                # Print full precision as requested
                self.logger.info(f"Component {axis} Best Validation MSE: {score}")
                metrics[axis] = score

        return metrics

    def predict(self, X):
        """
        Generates predictions for the input features.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            pd.DataFrame: A DataFrame containing 'azimuth' and 'zenith' columns, indexed by event_id.
        """
        preds = {}

        # Predict each component
        for axis in ["x", "y", "z"]:
            # Load model if not in memory
            if axis not in self.models:
                self._load_model(axis)

            model = self.models[axis]

            # Handle prediction based on model type (LGBMRegressor wrapper vs Booster)
            if isinstance(model, lgb.LGBMRegressor):
                preds[axis] = model.predict(X)
            elif isinstance(model, lgb.Booster):
                preds[axis] = model.predict(X)
            else:
                raise TypeError(f"Unknown model type for axis {axis}")

        # Reconstruct vector components
        px = preds["x"]
        py = preds["y"]
        pz = preds["z"]

        # Convert to spherical coordinates (azimuth, zenith)
        # The utility function handles vector normalization internally
        azimuth, zenith = cartesian_to_spherical(px, py, pz)

        # Return result as DataFrame
        return pd.DataFrame({"azimuth": azimuth, "zenith": zenith}, index=X.index)

    def _load_model(self, axis):
        """
        Loads a pre-trained model from disk.
        """
        path = self.model_paths[axis]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found for component {axis} at {path}"
            )

        self.logger.info(f"Loading model for {axis} from {path}")
        # Load as a Booster
        self.models[axis] = lgb.Booster(model_file=path)


def generate_submission(model, test_features, output_path):
    """
    Helper function to generate the submission file.

    Args:
        model (GradientBoostingVectorRegressor): The trained model.
        test_features (pd.DataFrame): Test set features (indexed by event_id).
        output_path (str): Path to save the CSV.
    """
    print(f"Generating predictions for {len(test_features)} test events...")

    # Generate predictions
    predictions = model.predict(test_features)

    # Reset index to make event_id a column
    submission = predictions.reset_index()

    # Ensure columns are in correct order: event_id, azimuth, zenith
    submission = submission[["event_id", "azimuth", "zenith"]]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
