import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import XGB_PARAMS, MIN_FARE, WORKING_DIR, SUBMISSION_FILE_PATH


class ResidualXGBRegressor:
    """
    XGBoost Regressor wrapper for Residual Learning.
    Predicts the residual (y - base_margin) given a base_margin.
    """

    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS.copy()
        self.model = None
        self.best_iteration = None

    def fit(
        self,
        X_train,
        y_train,
        train_margin,
        X_val=None,
        y_val=None,
        val_margin=None,
        verbose=True,
    ):
        """
        Trains the model using XGBoost with a base margin.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target (raw fare_amount).
            train_margin (pd.Series): Base margin for training (prior).
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation target.
            val_margin (pd.Series, optional): Base margin for validation.
            verbose (bool): Whether to print training progress.
        """
        # Create DMatrix for training
        # We explicitly set the base_margin so the model boosts from this baseline
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dtrain.set_base_margin(train_margin)

        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None and val_margin is not None:
            dval = xgb.DMatrix(data=X_val, label=y_val)
            dval.set_base_margin(val_margin)
            evals.append((dval, "validation"))

        # Extract training control parameters
        num_boost_round = self.params.get("n_estimators", 1000)
        early_stopping_rounds = self.params.get("early_stopping_rounds", 50)

        # Filter params for xgb.train (remove wrapper-specific params if any)
        train_params = {
            k: v
            for k, v in self.params.items()
            if k not in ["n_estimators", "early_stopping_rounds"]
        }

        if verbose:
            print(f"Starting training with params: {train_params}")

        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50 if verbose else False,
        )

        # Store best iteration (used for prediction)
        self.best_iteration = self.model.best_iteration

        if verbose:
            print(f"Training finished. Best iteration: {self.best_iteration}")
            if hasattr(self.model, "best_score"):
                print(f"Best Score: {self.model.best_score}")

    def predict(self, X, base_margin):
        """
        Predicts fare amount using the trained model and base margin.

        Args:
            X (pd.DataFrame): Features.
            base_margin (pd.Series): Base margin (prior).

        Returns:
            np.ndarray: Predicted fare amounts.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction.")

        dtest = xgb.DMatrix(data=X)
        dtest.set_base_margin(base_margin)

        # Use iteration_range to use the best model from early stopping
        # iteration_range=(start, end) -> [start, end)
        iteration_range = (
            (0, self.best_iteration + 1) if self.best_iteration is not None else None
        )

        # Predict
        # With base_margin set in DMatrix, raw prediction is margin + tree_sum
        preds = self.model.predict(dtest, iteration_range=iteration_range)

        # Post-processing: Apply floor and non-negative clamp
        preds = np.maximum(preds, MIN_FARE)

        return preds

    def save(self, filename="xgb_model.json"):
        """Saves the model to the working directory."""
        path = os.path.join(WORKING_DIR, filename)
        if self.model:
            self.model.save_model(path)
            print(f"Model saved to {path}")

    def load(self, filename="xgb_model.json"):
        """Loads the model from the working directory."""
        path = os.path.join(WORKING_DIR, filename)
        if os.path.exists(path):
            self.model = xgb.Booster()
            self.model.load_model(path)
            print(f"Model loaded from {path}")
        else:
            print(f"File {path} does not exist.")


def prepare_features(df, is_train=True):
    """
    Separates features, target, and margin from the dataframe.
    Drops non-feature columns like keys and timestamps.
    """
    # Columns to exclude from features
    exclude_cols = ["fare_amount", "base_margin", "key", "pickup_datetime"]

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    margin = df["base_margin"]

    y = None
    if is_train and "fare_amount" in df.columns:
        y = df["fare_amount"]

    return X, y, margin


def train_model(train_df, val_df, params=None):
    """
    Orchestrates the training process: prepares data and fits the ResidualXGBRegressor.
    """
    print("Preparing training data...")
    X_train, y_train, margin_train = prepare_features(train_df, is_train=True)

    print("Preparing validation data...")
    X_val, y_val, margin_val = prepare_features(val_df, is_train=True)

    model = ResidualXGBRegressor(params)
    model.fit(X_train, y_train, margin_train, X_val, y_val, margin_val)

    return model


def generate_submission(model, test_df, submission_path=SUBMISSION_FILE_PATH):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Preparing test data for prediction...")
    X_test, _, margin_test = prepare_features(test_df, is_train=False)
    keys = test_df["key"]

    print("Predicting...")
    predictions = model.predict(X_test, margin_test)

    print(f"Saving submission to {submission_path}...")
    submission = pd.DataFrame({"key": keys, "fare_amount": predictions})

    # Round to 2 decimals for currency format
    submission["fare_amount"] = submission["fare_amount"].round(2)

    submission.to_csv(submission_path, index=False)
    print("Submission saved successfully.")
