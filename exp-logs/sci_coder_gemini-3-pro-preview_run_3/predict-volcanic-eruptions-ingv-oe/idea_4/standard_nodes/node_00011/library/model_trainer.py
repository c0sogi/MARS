import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from library.config import (
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SUBMISSION_PATH,
)


class EruptionPredictor:
    """
    A wrapper class for the LightGBM Regressor to handle training and prediction
    specifically for the volcano eruption prediction task.
    """

    def __init__(self, params=None):
        """
        Initialize the predictor with LightGBM parameters.

        Args:
            params (dict, optional): Dictionary of LightGBM parameters.
                                     Defaults to LGBM_PARAMS from config.
        """
        self.params = params if params is not None else LGBM_PARAMS
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model using the provided training and validation data.
        Implements early stopping to prevent overfitting.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target (time_to_eruption).
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.
        """
        # Create LightGBM Datasets
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        # Prepare parameters and extract num_boost_round
        train_params = self.params.copy()
        # lgb.train uses 'num_boost_round' argument, defaulting to 100.
        # We extract 'n_estimators' from params to use as num_boost_round.
        num_boost_round = train_params.pop("n_estimators", 10000)

        # Define callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        print(f"Starting training with {num_boost_round} max rounds...")

        # Train the model
        self.model = lgb.train(
            params=train_params,
            train_set=train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[train_ds, val_ds],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Print full precision metrics for the best iteration
        print("\n--- Training Finished ---")
        print(f"Best Iteration: {self.model.best_iteration}")
        if self.model.best_score:
            for set_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    print(f"Best {set_name} {metric_name}: {score}")

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.array: Predicted time_to_eruption values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Predict using the best iteration found during training
        return self.model.predict(X, num_iteration=self.model.best_iteration)


def generate_submission(predictor, X_test, test_ids, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        predictor (EruptionPredictor): The trained model instance.
        X_test (pd.DataFrame): Test set features.
        test_ids (pd.Series): Segment IDs corresponding to the test set.
        output_path (str): Path to save the submission CSV.
    """
    print(f"Generating predictions for {len(X_test)} test samples...")

    predictions = predictor.predict(X_test)

    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": predictions}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
