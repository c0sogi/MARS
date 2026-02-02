import lightgbm as lgb
import numpy as np
import pickle
import os
from library.config import (
    LGBM_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    TARGET_COLS,
    SEED,
)
from library.utils import cartesian_to_spherical


class DirectionalLGBM:
    """
    A wrapper class for predicting neutrino direction using three independent
    LightGBM regressors, one for each Cartesian component (x, y, z).
    """

    def __init__(self, params=None):
        """
        Initialize the model with LightGBM parameters.

        Args:
            params (dict, optional): LightGBM parameters. If None, uses defaults from config.
        """
        self.params = params if params else LGBM_PARAMS.copy()
        self.models = {}
        self.targets = TARGET_COLS  # Expected: ["target_x", "target_y", "target_z"]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the three regressors (one per coordinate axis).

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets (N, 3) corresponding to [x, y, z].
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation targets (N, 3).
        """
        for i, target_name in enumerate(self.targets):
            print(f"\n{'='*20}")
            print(f"Training Regressor for Component: {target_name}")
            print(f"{'='*20}")

            # Extract specific component target
            y_tr_comp = y_train[:, i]

            # Create LightGBM Dataset
            train_ds = lgb.Dataset(X_train, label=y_tr_comp)

            valid_sets = [train_ds]
            valid_names = ["train"]

            if X_val is not None and y_val is not None:
                y_val_comp = y_val[:, i]
                val_ds = lgb.Dataset(X_val, label=y_val_comp, reference=train_ds)
                valid_sets.append(val_ds)
                valid_names.append("valid")

            # Train with early stopping
            # Note: We use the standard callback API for LightGBM
            callbacks = [
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=VERBOSE_EVAL),
            ]

            model = lgb.train(
                params=self.params,
                train_set=train_ds,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

            self.models[target_name] = model

    def predict(self, X):
        """
        Predicts azimuth and zenith angles for the input features.

        1. Predicts raw x, y, z components.
        2. Normalizes the resulting vector to unit length.
        3. Converts unit vector to spherical coordinates (azimuth, zenith).

        Args:
            X (np.ndarray): Input features.

        Returns:
            tuple: (azimuth, zenith) as numpy arrays.
        """
        # Ensure all models are trained
        for target in self.targets:
            if target not in self.models:
                raise ValueError(f"Model for {target} has not been trained.")

        # Predict components
        # Result is list of arrays: [pred_x, pred_y, pred_z]
        component_preds = [self.models[target].predict(X) for target in self.targets]

        # Stack to shape (N_samples, 3)
        pred_vectors = np.vstack(component_preds).T

        # Normalize vectors to unit length
        # Compute L2 norm for each row
        norms = np.linalg.norm(pred_vectors, axis=1, keepdims=True)

        # Handle zero vectors (though unlikely with regression outputs) to avoid NaN
        norms = np.where(norms == 0, 1e-9, norms)

        unit_vectors = pred_vectors / norms

        # Convert to Spherical Coordinates
        azimuth, zenith = cartesian_to_spherical(
            unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
        )

        return azimuth, zenith

    def save(self, file_path):
        """
        Saves the trained models to a pickle file.

        Args:
            file_path (str or Path): Destination path.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Models saved to {file_path}")

    def load(self, file_path):
        """
        Loads the models from a pickle file.

        Args:
            file_path (str or Path): Source path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")

        with open(file_path, "rb") as f:
            self.models = pickle.load(f)
        print(f"Models loaded from {file_path}")
