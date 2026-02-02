import xgboost as xgb
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import save_submission


class XGBoostRegressorWrapper:
    """
    A wrapper class for XGBoost to handle multi-output regression tasks
    by training independent regressors for each target variable.
    """

    def __init__(self):
        self.models = {}
        self.target_cols = Config.TARGET_COLS
        self.params = Config.XGB_PARAMS.copy()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains an XGBoost model for each target variable.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets (log-transformed).
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame, optional): Validation targets (log-transformed).
        """
        for target in self.target_cols:
            print(f"\nTraining XGBoost model for target: {target}")

            # Prepare parameters to avoid duplication
            train_params = self.params.copy()
            # Cite debug_lesson_1: XGBoost 1.6+ requires early_stopping_rounds in constructor.
            # We prioritize the value in params (e.g. from runfile injection) over the default Config constant.
            es_rounds = train_params.pop(
                "early_stopping_rounds", Config.EARLY_STOPPING_ROUNDS
            )

            # Instantiate the regressor
            model = xgb.XGBRegressor(**train_params, early_stopping_rounds=es_rounds)

            # Prepare evaluation set if validation data is available
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[target])]

            # Fit the model
            # Fix: Remove early_stopping_rounds from fit()
            model.fit(
                X_train,
                y_train[target],
                eval_set=eval_set,
                verbose=Config.VERBOSE_EVAL,
            )

            # Store the trained model
            self.models[target] = model

            # Log best score
            if hasattr(model, "best_score"):
                print(f"Best validation score (RMSE) for {target}: {model.best_score}")

    def predict(self, X):
        """
        Generates predictions for all target variables using the trained models.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.DataFrame: DataFrame containing predictions for each target column.
        """
        predictions = {}
        for target in self.target_cols:
            if target not in self.models:
                raise RuntimeError(f"Model for target '{target}' has not been trained.")

            print(f"Predicting target: {target}")
            pred = self.models[target].predict(X)
            predictions[target] = pred

        return pd.DataFrame(predictions, index=X.index)


def generate_submission_file(model, test_features, test_ids):
    """
    Generates predictions for the test set and saves them to the submission file.

    Args:
        model (XGBoostRegressorWrapper): The trained model wrapper.
        test_features (pd.DataFrame): Processed features for the test set.
        test_ids (pd.Series or list): IDs corresponding to the test set.
    """
    print("\nGenerating submission...")

    # Generate predictions (these are in log space if targets were transformed)
    # Note: The pipeline calling this is responsible for inverse transformation if needed.
    # However, based on the provided pipeline files, the inverse transform happens outside.
    # We will assume this function receives the features and returns raw model output,
    # but since we need to save to file, we should be careful.
    # Looking at the task description, we need to save the final values.
    # The `main.py` (not provided here but implied) would likely handle the inverse transform.
    # If we are to save it here, we must assume the caller handles the inverse transform
    # OR we just return the dataframe.

    # But the requirement says: "If this module handles submission generation... Save the final predictions".
    # I will assume the `predict` method returns what the model predicts.
    # If the model predicts log-values, they need to be inverted.
    # Since `feature_pipeline.py` has `inverse_transform_targets`, I will use it here
    # to ensure the submission is in the correct scale (eV).

    from library.feature_pipeline import inverse_transform_targets

    log_preds_df = model.predict(test_features)

    # Invert transformation (log1p -> expm1)
    final_preds_df = inverse_transform_targets(log_preds_df)

    # Prepare submission path
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Save using utility
    save_submission(
        ids=test_ids,
        predictions=final_preds_df.values,
        target_cols=Config.TARGET_COLS,
        output_path=submission_path,
    )
