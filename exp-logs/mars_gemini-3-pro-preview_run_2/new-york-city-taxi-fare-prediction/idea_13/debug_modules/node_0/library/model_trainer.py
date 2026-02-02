import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import (
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SUBMISSION_PATH,
    WORKING_DIR,
)


class ModelTrainer:
    """
    Handles the training of the XGBoost model and generation of predictions.
    """

    def __init__(self):
        self.model_path = os.path.join(WORKING_DIR, "xgb_model.json")
        self.model = None

    def train_xgboost(self, train_df, val_df):
        """
        Trains an XGBoost Regressor on the provided training data and evaluates on validation data.

        Args:
            train_df (pd.DataFrame): Training data containing features and 'fare_amount'.
            val_df (pd.DataFrame): Validation data containing features and 'fare_amount'.

        Returns:
            xgb.XGBRegressor: The trained model.
        """
        print("Initializing XGBoost Training...")

        # Define target and features
        target_col = "fare_amount"
        exclude_cols = [target_col, "key"]

        # Prepare Training Data
        X_train = train_df.drop(
            columns=[c for c in exclude_cols if c in train_df.columns]
        )
        y_train = train_df[target_col]

        # Prepare Validation Data
        X_val = val_df.drop(columns=[c for c in exclude_cols if c in val_df.columns])
        y_val = val_df[target_col]

        print(f"Training features: {list(X_train.columns)}")

        # Initialize Model
        # XGB_PARAMS contains 'n_estimators', 'learning_rate', 'objective', 'tree_method', 'device', etc.
        self.model = xgb.XGBRegressor(**XGB_PARAMS)

        # Fit Model with Early Stopping
        print(f"Fitting model with early_stopping_rounds={EARLY_STOPPING_ROUNDS}...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=VERBOSE_EVAL,
        )

        # Log Best Score
        # best_score refers to the best iteration's evaluation metric (RMSE)
        if hasattr(self.model, "best_score"):
            print(f"Best Validation RMSE: {self.model.best_score}")
        elif hasattr(self.model, "best_iteration"):
            # Fallback if best_score isn't directly accessible in this version wrapper
            # but usually accessible via evals_result if needed.
            # We rely on verbose output for the log, but let's try to print if possible.
            pass

        # Save Model
        print(f"Saving model to {self.model_path}...")
        self.model.save_model(self.model_path)

        return self.model

    def predict_and_postprocess(self, model, test_df):
        """
        Generates predictions for the test set and applies post-processing.

        Args:
            model (xgb.XGBRegressor): The trained model.
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            np.ndarray: Post-processed predictions.
        """
        print("Generating predictions on test set...")

        # Prepare Test Features (Exclude key)
        exclude_cols = [
            "key",
            "fare_amount",
        ]  # fare_amount shouldn't be there, but safety first
        X_test = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns])

        # Predict
        predictions = model.predict(X_test)

        # Post-Processing
        # Apply minimum fare floor of $2.50
        print("Applying post-processing (Minimum Fare Floor: $2.50)...")
        predictions = np.maximum(predictions, 2.50)

        return predictions

    def generate_submission(self, test_df, predictions):
        """
        Creates the submission CSV file.

        Args:
            test_df (pd.DataFrame): Test DataFrame containing the 'key' column.
            predictions (np.ndarray): The predicted fare amounts.
        """
        print(f"Generating submission file at {SUBMISSION_PATH}...")

        # Ensure submission directory exists (handled in config, but good practice)
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Save to CSV
        submission.to_csv(SUBMISSION_PATH, index=False)
        print("Submission file saved successfully.")
