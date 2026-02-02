import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
)
from library.preprocessing import get_preprocessed_data, inverse_transform_targets


class XGBoostRegressorWrapper:
    """
    Wrapper class for training and predicting with XGBoost models for multiple targets.
    """

    def __init__(self):
        self.models = {}
        self.params = XGB_PARAMS.copy()

    def train(self, train_df, val_df):
        """
        Trains separate XGBoost models for each target variable.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            val_df (pd.DataFrame): Validation data containing features and targets.
        """
        # Prepare feature matrices (drop non-feature columns)
        drop_cols = TARGET_COLS + ["id", "file_path"]
        X_train = train_df.drop(columns=drop_cols, errors="ignore")
        X_val = val_df.drop(columns=drop_cols, errors="ignore")

        print(
            f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples."
        )
        print(f"Features: {X_train.shape[1]}")

        for target in TARGET_COLS:
            print(f"\n--- Training model for target: {target} ---")
            y_train = train_df[target]
            y_val = val_df[target]

            model = xgb.XGBRegressor(**self.params)

            # Train with early stopping
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100,
            )

            self.models[target] = model

            # Evaluation
            # Note: Targets are already log1p transformed, so RMSE here is equivalent to RMSLE on original scale
            val_preds = model.predict(X_val)
            mse = mean_squared_error(y_val, val_preds)
            rmse = np.sqrt(mse)
            print(f"Validation RMSLE (log-space RMSE) for {target}: {rmse}")

    def predict(self, test_df):
        """
        Generates predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: DataFrame containing predictions for each target.
        """
        drop_cols = TARGET_COLS + ["id", "file_path"]
        X_test = test_df.drop(columns=drop_cols, errors="ignore")

        predictions = {}
        for target in TARGET_COLS:
            if target in self.models:
                predictions[target] = self.models[target].predict(X_test)
            else:
                raise RuntimeError(f"Model for target {target} has not been trained.")

        return pd.DataFrame(predictions)


def train_model(load_cached_data=True):
    """
    Orchestrates the data loading and model training process.
    """
    # Load preprocessed data
    # The preprocessing module handles caching and log-transform of targets
    train_df = get_preprocessed_data("train", load_cached_data=load_cached_data)
    val_df = get_preprocessed_data("val", load_cached_data=load_cached_data)

    # Initialize and train model wrapper
    model_wrapper = XGBoostRegressorWrapper()
    model_wrapper.train(train_df, val_df)

    return model_wrapper


def generate_submission(model_wrapper, load_cached_data=True):
    """
    Generates the submission file using the trained model.
    """
    print("\n--- Generating Submission ---")

    # Load test data
    test_df = get_preprocessed_data("test", load_cached_data=load_cached_data)

    # Generate predictions (in log space)
    log_preds_df = model_wrapper.predict(test_df)

    # Inverse transform predictions (expm1) to get back to original energy scale
    final_preds_df = log_preds_df.copy()
    for col in TARGET_COLS:
        final_preds_df[col] = inverse_transform_targets(log_preds_df[col].values)

    # Prepare submission dataframe
    # We use the sample submission to ensure correct ID order, though test_df should match
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # Ensure alignment by ID
    # We assume test_df has 'id' column preserved from metadata
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]

    for col in TARGET_COLS:
        submission[col] = final_preds_df[col]

    # Reorder to match sample submission just in case
    submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()

    # Save
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission.head())
