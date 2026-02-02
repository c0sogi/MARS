import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS


class LogTransformedXGBoost:
    """
    A wrapper for XGBoost Regressor that automatically applies a natural log
    transformation (log1p) to the target variable during training and an
    inverse transformation (expm1) during prediction.
    """

    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS
        # Initialize the XGBRegressor with provided parameters
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Trains the model on X and log1p(y).

        Args:
            X: Feature matrix.
            y: Target vector (raw values).
            eval_set: List of (X, y) tuples for validation (raw values).
            early_stopping_rounds: Activates early stopping.
            verbose: Logging verbosity.
        """
        # Transform training target
        y_log = np.log1p(y)

        # Transform validation sets if provided
        eval_set_log = []
        if eval_set:
            for X_val, y_val in eval_set:
                eval_set_log.append((X_val, np.log1p(y_val)))

        self.model.fit(
            X,
            y_log,
            eval_set=eval_set_log,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        return self

    def predict(self, X):
        """
        Predicts using the underlying model and applies expm1 to return to original scale.
        """
        pred_log = self.model.predict(X)
        # Apply inverse transformation
        return np.expm1(pred_log)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def best_iteration(self):
        return self.model.best_iteration


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    params=None,
    target_name="target",
    early_stopping_rounds=100,
    verbose=False,
):
    """
    Trains a LogTransformedXGBoost model and evaluates it.

    Args:
        X_train, X_val: Feature matrices.
        y_train, y_val: Target vectors (raw values).
        params: XGBoost hyperparameters.
        target_name: Name of the target for logging.
        early_stopping_rounds: Rounds for early stopping.
        verbose: Verbosity flag.

    Returns:
        Trained model instance.
    """
    print(f"Training model for {target_name}...")

    model = LogTransformedXGBoost(params=params)

    # Fit the model
    # Note: We pass raw y_train and y_val; the wrapper handles the log transform.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    # Generate predictions for evaluation (predictions are in raw space)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    # Calculate RMSLE (Root Mean Squared Logarithmic Error)
    # RMSLE is equivalent to RMSE of log-transformed values.
    train_rmsle = np.sqrt(mean_squared_error(np.log1p(y_train), np.log1p(train_preds)))
    val_rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(val_preds)))

    print(f"[{target_name}] Train RMSLE: {train_rmsle}")
    print(f"[{target_name}] Val RMSLE:   {val_rmsle}")

    return model


def generate_submission(
    model_formation, model_bandgap, X_test, test_ids, submission_path
):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model_formation: Trained model for formation energy.
        model_bandgap: Trained model for bandgap energy.
        X_test: Test features.
        test_ids: IDs for the test set.
        submission_path: Path to save the submission CSV.

    Returns:
        DataFrame containing the submission.
    """
    print("Generating predictions...")

    # Predict (models return raw values)
    pred_formation = model_formation.predict(X_test)
    pred_bandgap = model_bandgap.predict(X_test)

    # Create DataFrame
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "formation_energy_ev_natom": pred_formation,
            "bandgap_energy_ev": pred_bandgap,
        }
    )

    # Save
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission
