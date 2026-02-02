import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.config import (
    LGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SUBMISSION_PATH,
    SEED,
)


class EruptionPredictor:
    """
    Wrapper class for the LightGBM Regressor to predict time to eruption.
    """

    def __init__(self):
        # Initialize the model with parameters from config
        self.model = lgb.LGBMRegressor(**LGB_PARAMS)
        self.feature_names = None

    def _preprocess(self, X):
        """
        Internal utility to prepare the feature matrix.
        Removes 'segment_id' if present, as it is not a predictive feature.
        """
        if "segment_id" in X.columns:
            return X.drop(columns=["segment_id"])
        return X.copy()

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the model using the provided training and validation sets.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.
        """
        # Prepare data
        X_train_clean = self._preprocess(X_train)
        X_val_clean = self._preprocess(X_val)

        # Store feature names for consistency during prediction
        self.feature_names = X_train_clean.columns.tolist()

        # Define callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        # Train the model
        self.model.fit(
            X_train_clean,
            y_train,
            eval_set=[(X_val_clean, y_val)],
            eval_metric="mae",
            callbacks=callbacks,
        )

        # Print best validation score with full precision
        if self.model.best_score_:
            # The key might be 'l1' or 'mae' depending on the exact version/param combo
            # We iterate to find the metric score for the validation set
            for dataset_key, metrics in self.model.best_score_.items():
                for metric_name, score in metrics.items():
                    print(
                        f"Best Validation Score ({dataset_key} - {metric_name}): {score}"
                    )

    def predict(self, X_test):
        """
        Generates predictions for the given test data.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted time_to_eruption values.
        """
        X_test_clean = self._preprocess(X_test)

        # Ensure feature alignment
        if self.feature_names:
            # Reorder columns to match training data if necessary
            # (LightGBM is usually robust to col order if names match, but good practice)
            X_test_clean = X_test_clean[self.feature_names]

        return self.model.predict(X_test_clean)

    def create_submission(self, X_test, predictions):
        """
        Formats the predictions and saves them to the submission CSV file.

        Args:
            X_test (pd.DataFrame): Test DataFrame containing 'segment_id'.
            predictions (np.ndarray): Predicted values corresponding to X_test.
        """
        if "segment_id" not in X_test.columns:
            raise ValueError("X_test must contain 'segment_id' to generate submission.")

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "segment_id": X_test["segment_id"].astype(int),
                "time_to_eruption": predictions,
            }
        )

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
