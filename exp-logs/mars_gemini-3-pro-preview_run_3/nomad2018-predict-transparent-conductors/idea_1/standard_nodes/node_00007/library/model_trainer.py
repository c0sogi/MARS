import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from library.config import Config
from library.utils import log_transform, inverse_log_transform, calculate_rmsle


class XGBRegressorWrapper:
    """
    Wrapper for training and predicting with XGBoost Regression models for
    formation energy and bandgap energy.
    """

    def __init__(self):
        # Initialize independent models for each target
        # Applying shrinkage and stochastic regularization (Cite solution_lesson_node_00004)
        self.model_formation = XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=Config.RANDOM_SEED,
            objective="reg:squarederror",
        )
        self.model_bandgap = XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=Config.RANDOM_SEED,
            objective="reg:squarederror",
        )
        self.is_fitted = False

    def train(self, X_train, y_train):
        """
        Trains the XGBoost regression models.

        Args:
            X_train (pd.DataFrame or np.ndarray): Feature matrix.
            y_train (pd.DataFrame or np.ndarray): Target matrix with columns
                                                  [formation_energy, bandgap_energy].
        """
        # Ensure inputs are numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values

        # Apply log transformation if configured to match RMSLE objective
        if Config.APPLY_LOG_TARGET:
            y_train_trans = log_transform(y_train)
        else:
            y_train_trans = y_train

        print("Training Formation Energy Model (XGBoost)...")
        self.model_formation.fit(X_train, y_train_trans[:, 0])

        print("Training Bandgap Energy Model (XGBoost)...")
        self.model_bandgap.fit(X_train, y_train_trans[:, 1])

        self.is_fitted = True

        # Calculate training metrics on the training set itself
        y_pred_trans = np.column_stack(
            [self.model_formation.predict(X_train), self.model_bandgap.predict(X_train)]
        )

        if Config.APPLY_LOG_TARGET:
            y_pred = inverse_log_transform(y_pred_trans)
        else:
            y_pred = y_pred_trans

        # Ensure non-negative predictions as energies are physical quantities >= 0
        y_pred = np.maximum(y_pred, 0)

        train_rmsle = calculate_rmsle(y_train, y_pred)
        print(f"Training RMSLE: {train_rmsle}")

    def predict(self, X):
        """
        Generates predictions for the input features.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predictions with shape (n_samples, 2).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Predict log-transformed values
        pred_formation_trans = self.model_formation.predict(X)
        pred_bandgap_trans = self.model_bandgap.predict(X)

        # Stack predictions
        y_pred_trans = np.column_stack([pred_formation_trans, pred_bandgap_trans])

        # Inverse transform to get back to original scale
        if Config.APPLY_LOG_TARGET:
            y_pred = inverse_log_transform(y_pred_trans)
        else:
            y_pred = y_pred_trans

        # Clip negative values to 0
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def evaluate(self, X_val, y_val):
        """
        Evaluates the model on a validation set.

        Args:
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.DataFrame or np.ndarray): Validation targets.

        Returns:
            dict: Dictionary containing RMSLE scores.
        """
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.values

        y_pred = self.predict(X_val)

        rmsle = calculate_rmsle(y_val, y_pred)

        # Calculate individual RMSLE for each target for detailed feedback
        # Formation Energy is column 0
        rmsle_formation = calculate_rmsle(y_val[:, 0:1], y_pred[:, 0:1])
        # Bandgap Energy is column 1
        rmsle_bandgap = calculate_rmsle(y_val[:, 1:2], y_pred[:, 1:2])

        print(f"Validation RMSLE (Mean): {rmsle}")
        print(f"Validation RMSLE (Formation Energy): {rmsle_formation}")
        print(f"Validation RMSLE (Bandgap Energy): {rmsle_bandgap}")

        return {
            "rmsle_mean": rmsle,
            "rmsle_formation": rmsle_formation,
            "rmsle_bandgap": rmsle_bandgap,
        }
