import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import get_xgb_params, TARGET_COLS, RANDOM_SEED


class DualTargetRegressor:
    """
    Wrapper class that trains two separate XGBoost regressors for the two targets:
    1. Formation Energy (formation_energy_ev_natom)
    2. Bandgap Energy (bandgap_energy_ev)
    """

    def __init__(self, params=None):
        """
        Initialize the DualTargetRegressor.

        Args:
            params (dict, optional): XGBoost hyperparameters. If None, fetches from config.
        """
        self.params = params if params else get_xgb_params()

        # Initialize separate models for each target
        # We use the sklearn API of XGBoost for compatibility and ease of use
        self.model_formation = xgb.XGBRegressor(**self.params)
        self.model_bandgap = xgb.XGBRegressor(**self.params)

        self.targets = TARGET_COLS  # ["formation_energy_ev_natom", "bandgap_energy_ev"]

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds=100,
        verbose=False,
    ):
        """
        Trains both XGBoost models with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Log-transformed training targets.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame, optional): Log-transformed validation targets.
            early_stopping_rounds (int): Rounds for early stopping.
            verbose (bool): Whether to print XGBoost training logs.
        """
        print(f"Training DualTargetRegressor with params: {self.params}")

        # --- Train Formation Energy Model ---
        print(f"\n--- Training Target: {self.targets[0]} ---")
        y_train_form = y_train[self.targets[0]]

        eval_set_form = [(X_train, y_train_form)]
        if X_val is not None and y_val is not None:
            y_val_form = y_val[self.targets[0]]
            eval_set_form.append((X_val, y_val_form))

        self.model_formation.fit(
            X_train,
            y_train_form,
            eval_set=eval_set_form,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        # Report Validation Metric
        if X_val is not None and y_val is not None:
            preds_form = self.model_formation.predict(X_val)
            rmse_form = np.sqrt(mean_squared_error(y_val_form, preds_form))
            print(f"Validation RMSE for {self.targets[0]}: {rmse_form}")

        # --- Train Bandgap Energy Model ---
        print(f"\n--- Training Target: {self.targets[1]} ---")
        y_train_band = y_train[self.targets[1]]

        eval_set_band = [(X_train, y_train_band)]
        if X_val is not None and y_val is not None:
            y_val_band = y_val[self.targets[1]]
            eval_set_band.append((X_val, y_val_band))

        self.model_bandgap.fit(
            X_train,
            y_train_band,
            eval_set=eval_set_band,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        # Report Validation Metric
        if X_val is not None and y_val is not None:
            preds_band = self.model_bandgap.predict(X_val)
            rmse_band = np.sqrt(mean_squared_error(y_val_band, preds_band))
            print(f"Validation RMSE for {self.targets[1]}: {rmse_band}")

    def predict(self, X):
        """
        Generates predictions for both targets.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: DataFrame containing predicted values for both targets.
        """
        pred_form = self.model_formation.predict(X)
        pred_band = self.model_bandgap.predict(X)

        predictions = pd.DataFrame(
            {self.targets[0]: pred_form, self.targets[1]: pred_band}, index=X.index
        )

        return predictions
