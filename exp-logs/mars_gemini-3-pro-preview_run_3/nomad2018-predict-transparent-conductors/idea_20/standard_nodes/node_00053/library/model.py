import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import (
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    TARGET_COLS,
    SUBMISSION_PATH,
)
from library.data import inverse_transform_targets


class DualTargetRegressor:
    """
    Wrapper for training and predicting with two independent XGBoost models,
    one for formation energy and one for bandgap energy.
    """

    def __init__(self):
        # Initialize separate regressors for each target
        # Cite debug_lesson_1: Move early_stopping_rounds to constructor for XGBoost >= 1.6
        self.models = {
            target: xgb.XGBRegressor(
                **XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
            )
            for target in TARGET_COLS
        }

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains both models using the provided training and validation data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Log-transformed training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Log-transformed validation targets.

        Returns:
            dict: Dictionary containing validation MSE for each target.
        """
        metrics = {}

        for target in TARGET_COLS:
            print(f"\nTraining XGBoost model for target: {target}")
            model = self.models[target]

            # Prepare evaluation set for early stopping
            eval_set = [(X_train, y_train[target]), (X_val, y_val[target])]

            model.fit(
                X_train,
                y_train[target],
                eval_set=eval_set,
                verbose=VERBOSE_EVAL,
            )

            # Predict on validation set to calculate metric
            # Note: Predictions are in log space
            val_preds_log = model.predict(X_val)
            mse_log = mean_squared_error(y_val[target], val_preds_log)
            metrics[target] = mse_log

            print(f"Validation MSE (Log-Space) for {target}: {mse_log}")

        return metrics

    def predict(self, X):
        """
        Generates predictions for the input features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: DataFrame containing log-transformed predictions for both targets.
        """
        preds = {}
        for target in TARGET_COLS:
            # Predict returns numpy array
            preds[target] = self.models[target].predict(X)

        return pd.DataFrame(preds, index=X.index)


def save_submission(ids, log_preds_df, output_path=SUBMISSION_PATH):
    """
    Inverse transforms the log-scale predictions and saves them to a CSV file
    in the required submission format.

    Args:
        ids (array-like): Sequence of IDs corresponding to the predictions.
        log_preds_df (pd.DataFrame): DataFrame containing log-transformed predictions.
        output_path (str): Path to save the submission CSV.
    """
    # Apply inverse transformation: y = exp(z) - 1
    # We use the function from library.data to ensure consistency
    final_preds = inverse_transform_targets(log_preds_df)

    # Construct submission DataFrame
    submission = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": final_preds["formation_energy_ev_natom"],
            "bandgap_energy_ev": final_preds["bandgap_energy_ev"],
        }
    )

    # Save to CSV
    print(f"Saving submission to {output_path}...")
    submission.to_csv(output_path, index=False)
    print("Submission saved successfully.")
