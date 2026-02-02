import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    RANDOM_SEED,
    SUBMISSION_FILE,
    SAMPLE_SUBMISSION_CSV,
)
from library.data import (
    build_feature_matrix,
    log_transform_targets,
    inverse_transform_targets,
)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class DualTargetRegressor:
    """
    Regressor that trains separate XGBoost models for formation energy and bandgap energy.
    Handles log-transformation of targets internally during fit/predict.
    """

    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS.copy()
        self.models = {}
        self.feature_names = None

    def _prepare_data(self, df, is_training=True):
        """
        Separates features from metadata and targets.
        """
        # Columns to exclude from features
        exclude_cols = ["id", "file_path"] + TARGET_COLS

        # Identify feature columns
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = (
            df[TARGET_COLS]
            if is_training and set(TARGET_COLS).issubset(df.columns)
            else None
        )

        return X, y

    def fit(self, train_df, val_df=None, verbose=True):
        """
        Trains the models.
        """
        X_train, y_train_raw = self._prepare_data(train_df, is_training=True)
        self.feature_names = X_train.columns.tolist()

        # Log transform training targets
        train_df_log = log_transform_targets(train_df)
        y_train_log = train_df_log[TARGET_COLS]

        X_val = None
        y_val_log = None

        if val_df is not None:
            X_val, _ = self._prepare_data(val_df, is_training=True)
            val_df_log = log_transform_targets(val_df)
            y_val_log = val_df_log[TARGET_COLS]

        for target in TARGET_COLS:
            if verbose:
                print(f"Training model for target: {target}")

            model = xgb.XGBRegressor(**self.params)

            eval_set = []
            if X_val is not None:
                eval_set.append((X_val, y_val_log[target]))

            model.fit(
                X_train,
                y_train_log[target],
                eval_set=eval_set,
                early_stopping_rounds=100,
                verbose=False,
            )

            self.models[target] = model

            # Validation Metric Logging
            if X_val is not None:
                # Predict in log space
                preds_log = model.predict(X_val)
                # Calculate MSE in log space (which is equivalent to MSLE in original space)
                mse_log = mean_squared_error(y_val_log[target], preds_log)
                rmsle = np.sqrt(mse_log)
                print(f"Target {target} - Validation RMSLE: {rmsle}")

    def predict(self, test_df):
        """
        Generates predictions for the test set.
        """
        X_test, _ = self._prepare_data(test_df, is_training=False)

        # Align features with training data
        if self.feature_names:
            # Add missing columns with 0
            missing_cols = set(self.feature_names) - set(X_test.columns)
            for c in missing_cols:
                X_test[c] = 0
            # Ensure order and drop extra columns
            X_test = X_test[self.feature_names]

        predictions = {}
        for target in TARGET_COLS:
            if target in self.models:
                # Predict log(1+y)
                pred_log = self.models[target].predict(X_test)
                # Transform back to y
                pred_orig = inverse_transform_targets(pred_log)
                # Ensure no negative values (physical constraint)
                pred_orig = np.maximum(pred_orig, 0)
                predictions[target] = pred_orig

        return pd.DataFrame(predictions, index=test_df.index)


def train_and_predict(n_estimators=3000, subsample_size=None, load_cached_data=True):
    """
    Main pipeline execution: Load data, train model, evaluate, and generate submission.
    """
    print("Loading datasets...")
    train_df = build_feature_matrix("train", load_cached_data=load_cached_data)
    val_df = build_feature_matrix("val", load_cached_data=load_cached_data)
    test_df = build_feature_matrix("test", load_cached_data=load_cached_data)

    # Optional subsampling for debugging
    if subsample_size:
        print(f"Subsampling training data to {subsample_size} rows.")
        train_df = train_df.iloc[:subsample_size]

    # Configure parameters
    params = XGB_PARAMS.copy()
    params["n_estimators"] = n_estimators

    # Initialize and Train
    model = DualTargetRegressor(params)
    model.fit(train_df, val_df)

    # Final Evaluation on Validation Set
    print("\nOverall Validation Performance (RMSLE):")
    val_preds_df = model.predict(val_df)

    for target in TARGET_COLS:
        y_true = val_df[target]
        y_pred = val_preds_df[target]

        # RMSLE calculation
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
        print(f"{target}: {rmsle}")

    # Generate Submission
    print("\nGenerating submission...")
    test_preds_df = model.predict(test_df)

    # Construct submission dataframe preserving ID
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]
    for target in TARGET_COLS:
        submission[target] = test_preds_df[target].values

    # Save to file
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")
    print(submission.head())
