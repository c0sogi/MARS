import numpy as np
import lightgbm as lgb
from library.config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, VERBOSE_EVAL
from library.utils import cartesian_to_spherical, compute_angular_error


class VectorRegressor:
    """
    A regressor that predicts the 3D unit vector direction of a neutrino event
    using three independent LightGBM models (one for each Cartesian component).
    """

    def __init__(self):
        # Initialize three independent LightGBM regressors
        # We use a list to store models for x, y, and z components
        self.models = [
            lgb.LGBMRegressor(**LGBM_PARAMS),
            lgb.LGBMRegressor(**LGBM_PARAMS),
            lgb.LGBMRegressor(**LGBM_PARAMS),
        ]
        self.components = ["x", "y", "z"]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the three regressors on the provided data.

        Args:
            X_train (np.ndarray): Training features of shape (N_train, n_features).
            y_train (np.ndarray): Training targets (unit vectors) of shape (N_train, 3).
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation targets (unit vectors).
        """
        for i, component in enumerate(self.components):
            print(f"Training LightGBM model for component: target_{component}")

            # Extract the specific component target
            y_train_c = y_train[:, i]

            callbacks = []
            eval_set = None

            if X_val is not None and y_val is not None:
                y_val_c = y_val[:, i]
                eval_set = [(X_val, y_val_c)]

                # Configure callbacks for early stopping and logging
                # verbose=False in early_stopping to avoid redundant logs, log_evaluation handles it
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=VERBOSE_EVAL),
                ]

            # Train the model
            self.models[i].fit(
                X_train,
                y_train_c,
                eval_set=eval_set,
                eval_metric="mse",
                callbacks=callbacks,
            )

            # Print final validation MSE for this component if validation data exists
            if X_val is not None and y_val is not None:
                val_preds = self.models[i].predict(X_val)
                mse = np.mean((y_val_c - val_preds) ** 2)
                print(f"Final Validation MSE for target_{component}: {mse}")

    def predict(self, X):
        """
        Predict the normalized unit direction vector for the input features.

        Args:
            X (np.ndarray): Input features of shape (N, n_features).

        Returns:
            np.ndarray: Predicted unit vectors of shape (N, 3).
        """
        preds = []
        # Predict each component independently
        for model in self.models:
            preds.append(model.predict(X))

        # Stack predictions to form (N, 3) matrix
        vectors = np.column_stack(preds)

        # Compute vector magnitudes (norms)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Normalize vectors to unit length
        # Handle cases where norm is close to 0 to avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized_vectors = vectors / norms

        return normalized_vectors

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model using the competition metric (Mean Angular Error).

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets (unit vectors).

        Returns:
            float: The mean angular error in radians.
        """
        # Get predicted unit vectors
        pred_vectors = self.predict(X_val)

        # Convert predictions to spherical coordinates
        pred_azimuth, pred_zenith = cartesian_to_spherical(
            pred_vectors[:, 0], pred_vectors[:, 1], pred_vectors[:, 2]
        )

        # Convert true vectors to spherical coordinates
        true_azimuth, true_zenith = cartesian_to_spherical(
            y_val[:, 0], y_val[:, 1], y_val[:, 2]
        )

        # Compute angular error
        mae = compute_angular_error(
            true_azimuth, true_zenith, pred_azimuth, pred_zenith
        )

        print(f"Validation Mean Angular Error: {mae}")
        return mae
