import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.data import build_dataset, get_feature_target_split


class DualXGBoostRegressor:
    """
    Wrapper class for training two separate XGBoost models for formation energy
    and bandgap energy, handling log-transformation of targets.
    """

    def __init__(self, params=None):
        # Use default params from Config if not provided
        self.params = params if params else Config.XGB_PARAMS

        # Initialize two separate regressors
        self.model_formation = xgb.XGBRegressor(**self.params)
        self.model_bandgap = xgb.XGBRegressor(**self.params)

    def _log_transform(self, y):
        """Applies log(1+x) transformation to targets."""
        return np.log1p(y)

    def _inverse_transform(self, z):
        """Applies exp(x)-1 inverse transformation to predictions."""
        return np.expm1(z)

    def fit(self, train_df, val_df=None, early_stopping_rounds=100):
        """
        Trains both models on the provided data.
        """
        # Prepare training data
        X_train, y_train = get_feature_target_split(train_df)
        y_train_log = self._log_transform(y_train)

        # Prepare validation data if available
        eval_set_formation = None
        eval_set_bandgap = None

        if val_df is not None:
            X_val, y_val = get_feature_target_split(val_df)
            y_val_log = self._log_transform(y_val)

            # XGBoost expects eval_set as list of (X, y) tuples
            eval_set_formation = [
                (X_train, y_train_log["target_formation"]),
                (X_val, y_val_log["target_formation"]),
            ]
            eval_set_bandgap = [
                (X_train, y_train_log["target_bandgap"]),
                (X_val, y_val_log["target_bandgap"]),
            ]

        print("Training Formation Energy Model...")
        self.model_formation.fit(
            X_train,
            y_train_log["target_formation"],
            eval_set=eval_set_formation,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        print("Training Bandgap Energy Model...")
        self.model_bandgap.fit(
            X_train,
            y_train_log["target_bandgap"],
            eval_set=eval_set_bandgap,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        # Print validation metrics if validation set was provided
        if val_df is not None:
            self._print_metrics(val_df)

    def _print_metrics(self, val_df):
        """Calculates and prints RMSLE metrics on the validation set."""
        X_val, y_val = get_feature_target_split(val_df)

        # Predict on log scale
        pred_log_formation = self.model_formation.predict(X_val)
        pred_log_bandgap = self.model_bandgap.predict(X_val)

        # True values on log scale (for RMSLE calculation)
        y_val_log = self._log_transform(y_val)

        # Calculate RMSLE (RMSE of log-transformed values)
        rmsle_formation = np.sqrt(
            mean_squared_error(y_val_log["target_formation"], pred_log_formation)
        )
        rmsle_bandgap = np.sqrt(
            mean_squared_error(y_val_log["target_bandgap"], pred_log_bandgap)
        )
        mean_rmsle = (rmsle_formation + rmsle_bandgap) / 2

        print(f"Validation RMSLE Formation: {rmsle_formation}")
        print(f"Validation RMSLE Bandgap: {rmsle_bandgap}")
        print(f"Validation Mean RMSLE: {mean_rmsle}")

    def predict(self, test_df):
        """
        Generates predictions for the test set.
        Returns a DataFrame with columns: id, formation_energy_ev_natom, bandgap_energy_ev
        """
        X_test, _ = get_feature_target_split(test_df)

        # Predict log-transformed values
        pred_log_formation = self.model_formation.predict(X_test)
        pred_log_bandgap = self.model_bandgap.predict(X_test)

        # Inverse transform to original scale
        pred_formation = self._inverse_transform(pred_log_formation)
        pred_bandgap = self._inverse_transform(pred_log_bandgap)

        # Construct submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "formation_energy_ev_natom": pred_formation,
                "bandgap_energy_ev": pred_bandgap,
            }
        )

        return submission

    def save_submission(self, submission_df):
        """Saves the submission DataFrame to the configured path."""
        output_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run_pipeline(load_cached_data=True, debug=False):
    """
    Orchestrates the full training and inference pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed features from parquet.
        debug (bool): If True, runs on a small subset of data.
    """
    print("--- Starting Pipeline ---")

    # 1. Load Data
    print("Loading Training Data...")
    train_df = build_dataset("train", load_cached_data=load_cached_data, debug=debug)

    print("Loading Validation Data...")
    val_df = build_dataset("val", load_cached_data=load_cached_data, debug=debug)

    print("Loading Test Data...")
    test_df = build_dataset("test", load_cached_data=load_cached_data, debug=debug)

    # 2. Initialize Model
    model = DualXGBoostRegressor()

    # 3. Train Model
    print("Training Models...")
    # Adjust early stopping rounds based on debug mode
    es_rounds = 10 if debug else 100
    model.fit(train_df, val_df, early_stopping_rounds=es_rounds)

    # 4. Generate Predictions
    print("Generating Predictions...")
    submission_df = model.predict(test_df)

    # 5. Save Submission
    model.save_submission(submission_df)

    print("--- Pipeline Completed ---")
