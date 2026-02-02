import os
import pandas as pd
import numpy as np
import xgboost as xgb
from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_CSV,
)
from library.utils import log_transform, inverse_log_transform, calculate_rmsle


class GradientBoostingPredictor:
    """
    Wrapper for XGBoost Regressor to handle multi-output regression
    with log-transformed targets and element-specific feature handling.
    """

    def __init__(self):
        self.models = {}
        self.feature_cols = []

    def fit_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains separate XGBoost models for each target variable.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            val_df (pd.DataFrame): Validation data containing features and targets.
        """
        # Identify feature columns (exclude metadata like id and targets)
        exclude_cols = ["id", "file_path"] + TARGET_COLS
        self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        print(f"Training on {len(self.feature_cols)} features.")

        # Prepare validation sets
        X_train = train_df[self.feature_cols]
        X_val = val_df[self.feature_cols]

        overall_rmsle = 0.0

        for target in TARGET_COLS:
            print(f"\n--- Training model for target: {target} ---")

            # Extract and log-transform targets
            y_train = train_df[target]
            y_val = val_df[target]

            y_train_log = log_transform(y_train)
            y_val_log = log_transform(y_val)

            # Initialize XGBoost Regressor
            # Cite debug_lesson_1: Move early_stopping_rounds to constructor for XGBoost > 1.6
            model = xgb.XGBRegressor(early_stopping_rounds=100, **XGB_PARAMS)

            # Train with early stopping
            # Note: XGBoost handles NaNs in features automatically
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
                verbose=250,  # Print progress every 250 rounds
            )

            self.models[target] = model

            # Evaluate on validation set
            # Predict log values then inverse transform
            preds_log = model.predict(X_val)
            preds = inverse_log_transform(preds_log)

            # Calculate metric
            score = calculate_rmsle(y_val.values, preds)
            print(f"Validation RMSLE for {target}: {score}")
            overall_rmsle += score

        avg_rmsle = overall_rmsle / len(TARGET_COLS)
        print(f"\nAverage Validation RMSLE: {avg_rmsle}")

    def predict_values(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the test set using trained models.

        Args:
            test_df (pd.DataFrame): Test data features.

        Returns:
            pd.DataFrame: DataFrame containing predictions for all targets.
        """
        if not self.models:
            raise RuntimeError("Models have not been trained. Call fit_model first.")

        X_test = test_df[self.feature_cols]
        predictions = {}

        for target in TARGET_COLS:
            model = self.models[target]

            # Predict (returns log scale)
            preds_log = model.predict(X_test)

            # Inverse transform to original scale
            preds = inverse_log_transform(preds_log)

            # Ensure non-negative predictions (physical constraint)
            preds = np.maximum(preds, 0)

            predictions[target] = preds

        return pd.DataFrame(predictions)

    def generate_submission(self, test_df: pd.DataFrame):
        """
        Generates the submission file.

        Args:
            test_df (pd.DataFrame): Test data features including 'id'.
        """
        print("\nGenerating submission...")

        # Generate predictions
        preds_df = self.predict_values(test_df)
        preds_df["id"] = test_df["id"].values

        # Load sample submission to ensure correct format and order
        if os.path.exists(SAMPLE_SUBMISSION_CSV):
            submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)

            # Update values in sample submission based on ID matching
            # We set index to ID to ensure alignment
            submission.set_index("id", inplace=True)
            preds_df.set_index("id", inplace=True)

            submission.update(preds_df)
            submission.reset_index(inplace=True)
        else:
            # Fallback if sample submission missing
            submission = preds_df[["id"] + TARGET_COLS]

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
