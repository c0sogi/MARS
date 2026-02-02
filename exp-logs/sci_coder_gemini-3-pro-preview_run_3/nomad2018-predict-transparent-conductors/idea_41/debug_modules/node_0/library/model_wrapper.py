import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, EARLY_STOPPING_ROUNDS, VERBOSE_EVAL
from library.preprocessor import TargetTransformer


class XGBRegressorWrapper:
    """
    Wrapper for training and predicting with XGBoost models for multiple targets.
    Handles target transformation (log1p) and inverse transformation (expm1) internally.
    """

    def __init__(self):
        # Initialize separate models for each target
        self.models = {
            "formation_energy_ev_natom": xgb.XGBRegressor(**XGB_PARAMS),
            "bandgap_energy_ev": xgb.XGBRegressor(**XGB_PARAMS),
        }
        self.transformer = TargetTransformer()
        self.targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost models with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Validation targets.

        Returns:
            dict: Validation RMSLE for each target.
        """
        print("Starting training of XGBoost models...")
        metrics = {}

        for target in self.targets:
            print(f"\n--- Training model for target: {target} ---")

            # Apply log(1+x) transformation to targets
            # This stabilizes variance and matches the RMSLE metric (which is RMSE on log data)
            y_train_trans = self.transformer.transform(y_train[target])
            y_val_trans = self.transformer.transform(y_val[target])

            model = self.models[target]

            # Fit model with early stopping
            model.fit(
                X_train,
                y_train_trans,
                eval_set=[(X_train, y_train_trans), (X_val, y_val_trans)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=VERBOSE_EVAL,
            )

            # Evaluate on validation set
            # Predict returns values in log space
            val_preds_log = model.predict(X_val)

            # Calculate RMSE on log-transformed data, which is equivalent to RMSLE on original data
            val_rmsle = np.sqrt(mean_squared_error(y_val_trans, val_preds_log))

            print(f"Validation RMSLE for {target}: {val_rmsle:.6f}")
            metrics[target] = val_rmsle

        avg_rmsle = np.mean(list(metrics.values()))
        print(f"\nAverage Validation RMSLE: {avg_rmsle:.6f}")

        return metrics

    def predict(self, X):
        """
        Generates predictions for the input features.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            pd.DataFrame: DataFrame containing predictions for both targets in original units.
        """
        predictions = {}

        for target in self.targets:
            model = self.models[target]

            # Generate predictions (in log space)
            pred_log = model.predict(X)

            # Inverse transform to get original units (eV/atom or eV)
            pred_orig = self.transformer.inverse_transform(pred_log)

            predictions[target] = pred_orig

        return pd.DataFrame(predictions, index=X.index)
