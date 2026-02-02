import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, TARGET_COLS, SUBMISSION_PATH
from library.preprocessor import TargetTransformer


class DualTargetRegressor:
    """
    Wraps two XGBoost regressors to predict formation energy and bandgap energy separately.
    Handles target transformation (log1p) internally to optimize for RMSLE.
    """

    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS
        self.models = {}
        self.transformers = {}
        self.targets = TARGET_COLS

        # Initialize models and transformers for each target
        for target in self.targets:
            self.models[target] = xgb.XGBRegressor(**self.params)
            self.transformers[target] = TargetTransformer()

    def fit(
        self, X_train, y_train, X_val, y_val, early_stopping_rounds=100, verbose=False
    ):
        """
        Trains the models on the provided data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Validation targets.
            early_stopping_rounds (int): Rounds for early stopping.
            verbose (bool): Whether to print training logs.
        """
        print("Starting training for DualTargetRegressor...")

        for target in self.targets:
            print(f"\n--- Training model for target: {target} ---")

            # Prepare data
            y_train_vec = y_train[target].values
            y_val_vec = y_val[target].values

            # Transform targets (log1p)
            # This converts the regression task to minimize MSLE effectively
            y_train_trans = self.transformers[target].transform(y_train_vec)
            y_val_trans = self.transformers[target].transform(y_val_vec)

            # Train
            self.models[target].set_params(early_stopping_rounds=early_stopping_rounds)
            self.models[target].fit(
                X_train,
                y_train_trans,
                eval_set=[(X_train, y_train_trans), (X_val, y_val_trans)],
                verbose=100 if verbose else False,
            )

            # Report best score
            best_score = self.models[target].best_score
            print(f"Best validation score (RMSE on log-transformed data): {best_score}")

    def predict(self, X):
        """
        Predicts targets for the given features.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            pd.DataFrame: Predicted values for both targets.
        """
        predictions = {}

        for target in self.targets:
            # Predict in transformed space
            y_pred_trans = self.models[target].predict(X)

            # Inverse transform to original space
            y_pred = self.transformers[target].inverse_transform(y_pred_trans)

            # Ensure non-negative predictions for physical energies (sanity check)
            y_pred = np.maximum(y_pred, 0.0)

            predictions[target] = y_pred

        return pd.DataFrame(predictions)

    def evaluate(self, X_val, y_val):
        """
        Evaluates the model using RMSLE (Root Mean Squared Logarithmic Error).
        Since we trained on log1p targets, RMSE on transformed predictions matches RMSLE on original data.
        """
        print("\n--- Model Evaluation (RMSLE) ---")
        metrics = {}

        for target in self.targets:
            y_true = y_val[target].values

            # Get predictions (already inverse transformed by predict method)
            preds_df = self.predict(X_val)
            y_pred = preds_df[target].values

            # Calculate RMSLE manually to be sure
            # RMSLE = sqrt(mean((log(1+p) - log(1+a))^2))
            log_true = np.log1p(y_true)
            log_pred = np.log1p(y_pred)
            mse = mean_squared_error(log_true, log_pred)
            rmsle = np.sqrt(mse)

            metrics[target] = rmsle
            print(f"{target}: {rmsle}")

        # Calculate mean RMSLE across columns
        mean_rmsle = np.mean(list(metrics.values()))
        print(f"Average Column-wise RMSLE: {mean_rmsle}")

        return metrics


def generate_submission(regressor, X_test, test_ids, output_path=SUBMISSION_PATH):
    """
    Generates the submission file using the trained regressor.

    Args:
        regressor (DualTargetRegressor): Trained model instance.
        X_test (pd.DataFrame): Test features.
        test_ids (pd.Series or list): IDs corresponding to the test features.
        output_path (str): Path to save the CSV.
    """
    print(f"\nGenerating submission file at {output_path}...")

    # Generate predictions
    preds_df = regressor.predict(X_test)

    # Combine with IDs
    submission_df = pd.DataFrame({"id": test_ids})

    # Add target columns
    for target in TARGET_COLS:
        submission_df[target] = preds_df[target]

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print("Submission file saved successfully.")
    print(submission_df.head())
