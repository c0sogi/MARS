import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

from library.config import XGB_PARAMS, TARGET_COLS, SUBMISSION_PATH
from library.data_loader import load_and_process_data, inverse_transform_targets


class DualTargetRegressor:
    """
    Wrapper class to manage two independent regressors for formation energy and bandgap energy.
    """

    def __init__(self, params=None):
        # Use provided params or default to global config
        # We copy to avoid modifying the global dictionary
        self.params = params.copy() if params else XGB_PARAMS.copy()

        # Extract fit-time parameters if present to avoid constructor warnings/errors
        self.fit_params = {}
        # Cite {debug_lesson_1}: early_stopping_rounds should be passed to constructor in XGBoost 1.6+
        # We leave it in self.params so it is passed to XGBRegressor(**self.params)

        self.models = {}
        for target in TARGET_COLS:
            self.models[target] = xgb.XGBRegressor(**self.params)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains both regressors on the provided data.
        """
        for target in TARGET_COLS:
            print(f"\n[DualTargetRegressor] Training model for target: {target}")
            model = self.models[target]

            # Prepare evaluation set for early stopping
            eval_set = [(X_train, y_train[target]), (X_val, y_val[target])]

            model.fit(
                X_train,
                y_train[target],
                eval_set=eval_set,
                verbose=False,  # Suppress iteration logs
                **self.fit_params,
            )

            # Evaluation on validation set
            # Predictions are in log space (z = log(1+y))
            preds_log = model.predict(X_val)
            rmse_log = np.sqrt(mean_squared_error(y_val[target], preds_log))

            # Print full precision metric
            print(f"Validation RMSLE (log-scale RMSE) for {target}: {rmse_log}")

            if hasattr(model, "best_iteration"):
                print(f"Best iteration: {model.best_iteration}")

    def predict(self, X):
        """
        Generates predictions for both targets and transforms them back to the original scale.
        """
        predictions = {}
        for target in TARGET_COLS:
            model = self.models[target]
            # Predict log-transformed values
            pred_log = model.predict(X)
            predictions[target] = pred_log

        # Convert to DataFrame
        pred_df_log = pd.DataFrame(predictions, index=X.index)

        # Inverse transform to original scale: y = exp(z) - 1
        pred_df_original = inverse_transform_targets(pred_df_log)

        return pred_df_original


def train_model(load_cached_data=True, max_samples=None):
    """
    Orchestrates data loading and model training.

    Args:
        load_cached_data (bool): If True, tries to load pre-computed features from parquet.
        max_samples (int): Optional limit on number of samples for debugging.

    Returns:
        DualTargetRegressor: The trained model instance.
    """
    print(f"Loading training data (cached={load_cached_data})...")
    X_train, y_train = load_and_process_data(
        dataset_type="train", load_cached_data=load_cached_data, max_samples=max_samples
    )

    print(f"Loading validation data (cached={load_cached_data})...")
    X_val, y_val = load_and_process_data(
        dataset_type="val", load_cached_data=load_cached_data, max_samples=max_samples
    )

    # Initialize model
    model = DualTargetRegressor()

    # Fit model
    model.fit(X_train, y_train, X_val, y_val)

    return model


def generate_submission(model, load_cached_data=True):
    """
    Loads test data, generates predictions, and saves the submission file.

    Args:
        model (DualTargetRegressor): Trained model.
        load_cached_data (bool): If True, tries to load pre-computed features from parquet.
    """
    print(f"Loading test data (cached={load_cached_data})...")
    X_test, ids = load_and_process_data(
        dataset_type="test", load_cached_data=load_cached_data
    )

    print("Generating predictions on test set...")
    preds_df = model.predict(X_test)

    # Prepare submission DataFrame
    # Ensure columns are in the correct order: id, formation_energy_ev_natom, bandgap_energy_ev
    submission = pd.DataFrame()
    submission["id"] = ids
    for target in TARGET_COLS:
        submission[target] = preds_df[target].values

    # Save
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
