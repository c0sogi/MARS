import lightgbm as lgb
import numpy as np
import pandas as pd
from library.config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS


class VolcanoLGBM:
    """
    Wrapper class for the LightGBM Regressor to handle training and prediction
    for the volcano eruption prediction task.
    """

    def __init__(self, params=None):
        """
        Initialize the model with specific parameters.

        Args:
            params (dict, optional): Dictionary of LightGBM parameters.
                                     Defaults to library.config.LGBM_PARAMS.
        """
        # Use provided params or fall back to the global configuration
        self.params = params if params is not None else LGBM_PARAMS.copy()
        self.model = None

    def train(self, X_train, y_train, X_val, y_val, verbose_eval=100):
        """
        Trains the LightGBM model using the provided training and validation sets.
        Implements early stopping based on the validation metric.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training targets.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation targets.
            verbose_eval (int): Frequency of logging training metrics.
        """
        # Create LightGBM datasets
        # Reference the training set in the validation set to ensure consistent binning
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        # Configure callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=verbose_eval),
        ]

        # Train the model
        # Note: n_estimators is included in self.params (from config)
        self.model = lgb.train(
            params=self.params,
            train_set=train_ds,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Explicitly print the best score with full precision
        if self.model.best_score:
            # The metric is 'mae', which LightGBM typically stores as 'l1' or 'mae'
            valid_scores = self.model.best_score.get("valid", {})
            # Check for common keys for Mean Absolute Error
            mae = valid_scores.get("l1") or valid_scores.get("mae")

            if mae is not None:
                print(f"Final Best Validation MAE: {mae}")
            else:
                # Fallback if key is different
                print(f"Final Best Validation Scores: {valid_scores}")

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Array of predicted time_to_eruption values.
        """
        if self.model is None:
            raise RuntimeError("The model must be trained before making predictions.")

        # Predict using the best iteration found during training
        return self.model.predict(X, num_iteration=self.model.best_iteration)
