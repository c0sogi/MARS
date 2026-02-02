import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os
import library.config as config


class EnergyPredictor:
    """
    Handles the training and prediction of Formation Energy and Bandgap Energy
    using separate XGBoost regressors.
    """

    def __init__(self):
        # Initialize models with parameters from config
        self.model_formation = xgb.XGBRegressor(
            **config.XGB_PARAMS, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS
        )
        self.model_bandgap = xgb.XGBRegressor(
            **config.XGB_PARAMS, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS
        )
        self.targets = config.TARGET_COLS
        self.feature_cols = None

    def _prepare_data(self, df, is_training=True):
        """
        Prepares feature matrix X and target vector y (if training).
        Handles feature alignment and log-transformation of targets.
        """
        # Columns to exclude from features
        exclude = set(["id", "file_path"] + self.targets)

        # Determine feature columns during the first training call
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c not in exclude]
            self.feature_cols.sort()  # Ensure consistent order

        # Handle missing columns in test set (e.g., if specific element pairs weren't present)
        # We fill them with 0.0 as they represent counts/histograms/properties of missing elements.
        if not is_training:
            missing_cols = set(self.feature_cols) - set(df.columns)
            for c in missing_cols:
                df[c] = 0.0

        # Select features in the correct order
        X = df[self.feature_cols]

        y = None
        if is_training:
            if not all(t in df.columns for t in self.targets):
                raise ValueError(
                    f"Target columns {self.targets} missing in training data."
                )

            # Extract targets
            y_raw = df[self.targets]

            # Apply Log(1+x) transformation
            # Clip to 0 to prevent log of negative numbers (though energy shouldn't be negative here ideally)
            y_raw = y_raw.clip(lower=0)
            y = np.log1p(y_raw)

        return X, y

    def train(self, train_df, val_df):
        """
        Trains the XGBoost models with Early Stopping.
        """
        print(
            f"Training on {len(train_df)} samples, validating on {len(val_df)} samples."
        )

        X_train, y_train = self._prepare_data(train_df, is_training=True)
        X_val, y_val = self._prepare_data(val_df, is_training=True)

        # --- 1. Train Formation Energy Model ---
        target_form = self.targets[0]
        print(f"\n--- Training {target_form} Model ---")

        self.model_formation.fit(
            X_train,
            y_train[target_form],
            eval_set=[(X_train, y_train[target_form]), (X_val, y_val[target_form])],
            verbose=500,
        )

        # Evaluate Formation Energy
        val_preds_log_form = self.model_formation.predict(X_val)
        mse_form = mean_squared_error(y_val[target_form], val_preds_log_form)
        rmsle_form = np.sqrt(mse_form)
        print(f"{target_form} Validation RMSLE: {rmsle_form}")

        # --- 2. Train Bandgap Energy Model ---
        target_band = self.targets[1]
        print(f"\n--- Training {target_band} Model ---")

        self.model_bandgap.fit(
            X_train,
            y_train[target_band],
            eval_set=[(X_train, y_train[target_band]), (X_val, y_val[target_band])],
            verbose=500,
        )

        # Evaluate Bandgap Energy
        val_preds_log_band = self.model_bandgap.predict(X_val)
        mse_band = mean_squared_error(y_val[target_band], val_preds_log_band)
        rmsle_band = np.sqrt(mse_band)
        print(f"{target_band} Validation RMSLE: {rmsle_band}")

        # Average RMSLE
        avg_rmsle = (rmsle_form + rmsle_band) / 2
        print(f"\nAverage RMSLE: {avg_rmsle}")

    def predict(self, test_df):
        """
        Generates predictions for the test set.
        Applies inverse log transformation.
        """
        print(f"\nGenerating predictions for {len(test_df)} samples...")

        X_test, _ = self._prepare_data(test_df, is_training=False)

        # Predict in log space
        pred_log_form = self.model_formation.predict(X_test)
        pred_log_band = self.model_bandgap.predict(X_test)

        # Inverse transform: exp(y) - 1
        pred_form = np.expm1(pred_log_form)
        pred_band = np.expm1(pred_log_band)

        # Ensure non-negative predictions
        pred_form = np.maximum(pred_form, 0)
        pred_band = np.maximum(pred_band, 0)

        # Construct submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                self.targets[0]: pred_form,
                self.targets[1]: pred_band,
            }
        )

        return submission


def save_submission(submission_df):
    """
    Saves the submission DataFrame to the configured path.
    """
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
