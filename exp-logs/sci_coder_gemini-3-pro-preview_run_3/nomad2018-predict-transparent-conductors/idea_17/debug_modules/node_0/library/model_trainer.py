import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config


class DualTargetRegressor:
    """
    Manages training and prediction for two independent XGBoost regressors:
    1. Formation Energy (eV/atom)
    2. Bandgap Energy (eV)

    Handles logarithmic target transformation (log1p) and inverse transformation (expm1)
    to optimize for Root Mean Squared Logarithmic Error (RMSLE).
    """

    def __init__(self):
        self.config = Config
        self.model_formation = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        self.model_bandgap = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        self.feature_cols = None
        self.targets = (
            self.config.TARGET_COLS
        )  # ["formation_energy_ev_natom", "bandgap_energy_ev"]

    def _transform_target(self, y):
        """
        Applies log(1 + y) transformation.
        """
        return np.log1p(y)

    def _inverse_transform_target(self, z):
        """
        Applies exp(z) - 1 transformation.
        Ensures non-negative predictions.
        """
        return np.expm1(z)

    def _get_features(self, df):
        """
        Extracts feature columns, excluding metadata and targets.
        """
        # Exclude ID, file_path, and targets
        exclude = {"id", "file_path"} | set(self.targets)

        # If feature_cols is already defined (during prediction), use it to ensure order
        if self.feature_cols is not None:
            # Ensure all expected columns exist, fill missing with 0 if necessary (though data processor handles this)
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                # In a strict pipeline, this shouldn't happen, but for robustness:
                for c in missing:
                    df[c] = 0.0
            return df[self.feature_cols]

        # Otherwise, determine features from dataframe
        cols = [c for c in df.columns if c not in exclude]
        # Filter for numeric types just in case
        return df[cols].select_dtypes(include=[np.number])

    def fit(self, train_df, val_df):
        """
        Trains both XGBoost models using the provided training and validation data.
        """
        print("Starting training...")

        # Determine feature columns from training data
        X_train = self._get_features(train_df)
        self.feature_cols = X_train.columns.tolist()
        print(f"Training with {len(self.feature_cols)} features.")

        X_val = self._get_features(val_df)

        # Prepare Targets (Log Transformed)
        y_train_form = self._transform_target(train_df[self.targets[0]])
        y_train_band = self._transform_target(train_df[self.targets[1]])

        y_val_form = self._transform_target(val_df[self.targets[0]])
        y_val_band = self._transform_target(val_df[self.targets[1]])

        # --- Train Formation Energy Model ---
        print(f"\nTraining Model 1: {self.targets[0]} (Formation Energy)")
        self.model_formation.fit(
            X_train,
            y_train_form,
            eval_set=[(X_train, y_train_form), (X_val, y_val_form)],
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            verbose=self.config.VERBOSE_EVAL,
        )

        # --- Train Bandgap Energy Model ---
        print(f"\nTraining Model 2: {self.targets[1]} (Bandgap Energy)")
        self.model_bandgap.fit(
            X_train,
            y_train_band,
            eval_set=[(X_train, y_train_band), (X_val, y_val_band)],
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            verbose=self.config.VERBOSE_EVAL,
        )

        print("\nTraining completed.")

    def predict(self, test_df):
        """
        Generates predictions for the test set.
        Returns a DataFrame with columns: [id, formation_energy_ev_natom, bandgap_energy_ev]
        """
        if self.feature_cols is None:
            raise ValueError("Model has not been trained yet.")

        X_test = self._get_features(test_df)

        # Predict in log space
        pred_log_form = self.model_formation.predict(X_test)
        pred_log_band = self.model_bandgap.predict(X_test)

        # Inverse transform
        pred_form = self._inverse_transform_target(pred_log_form)
        pred_band = self._inverse_transform_target(pred_log_band)

        # Ensure non-negative (physics constraint)
        pred_form = np.maximum(0, pred_form)
        pred_band = np.maximum(0, pred_band)

        # Construct result DataFrame
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                self.targets[0]: pred_form,
                self.targets[1]: pred_band,
            }
        )

        return submission

    def evaluate(self, val_df):
        """
        Evaluates the model on the validation set and prints RMSLE scores.
        """
        print("\n--- Evaluation on Validation Set ---")

        # Generate predictions
        preds_df = self.predict(val_df)

        # Align indices (merge on ID to be safe)
        merged = pd.merge(
            val_df[["id"] + self.targets],
            preds_df,
            on="id",
            suffixes=("_true", "_pred"),
        )

        # Calculate RMSLE for Formation Energy
        # RMSLE = sqrt(mean_squared_error(log1p(true), log1p(pred)))
        y_true_form = merged[f"{self.targets[0]}_true"]
        y_pred_form = merged[f"{self.targets[0]}_pred"]
        rmsle_form = np.sqrt(
            mean_squared_error(
                self._transform_target(y_true_form), self._transform_target(y_pred_form)
            )
        )

        # Calculate RMSLE for Bandgap Energy
        y_true_band = merged[f"{self.targets[1]}_true"]
        y_pred_band = merged[f"{self.targets[1]}_pred"]
        rmsle_band = np.sqrt(
            mean_squared_error(
                self._transform_target(y_true_band), self._transform_target(y_pred_band)
            )
        )

        mean_rmsle = (rmsle_form + rmsle_band) / 2

        print(f"RMSLE Formation Energy: {rmsle_form}")
        print(f"RMSLE Bandgap Energy:   {rmsle_band}")
        print(f"Mean RMSLE:             {mean_rmsle}")

        return mean_rmsle

    def save_submission(self, submission_df):
        """
        Saves the submission DataFrame to the configured path.
        """
        path = self.config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        submission_df.to_csv(path, index=False)
        print(f"Submission saved to {path}")
