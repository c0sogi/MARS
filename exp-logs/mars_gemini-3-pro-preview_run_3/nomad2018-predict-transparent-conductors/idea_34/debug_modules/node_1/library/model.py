import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    WORKING_DIR,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
    TEST_METADATA_PATH,
)
from library.data_loader import load_data, inverse_transform_targets


class DualTargetRegressor:
    """
    Wrapper class to handle two separate XGBoost regressors for the two targets.
    """

    def __init__(self, params=None):
        self.params = params if params else {}
        self.models = {}
        for target in TARGET_COLS:
            self.models[target] = xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds=100,
        verbose=False,
    ):
        """
        Fits both models (one per target).
        """
        for target in TARGET_COLS:
            print(f"\nTraining model for target: {target}")
            model = self.models[target]

            # Prepare eval set if validation data is provided
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train[target]), (X_val, y_val[target])]

            # Cite debug_lesson_1: Set early_stopping_rounds via set_params for XGBoost 1.6+
            # Cite debug_lesson_15: Explicitly disable early stopping if no eval set is provided
            if eval_set:
                model.set_params(early_stopping_rounds=early_stopping_rounds)
            else:
                model.set_params(early_stopping_rounds=None)

            model.fit(
                X_train,
                y_train[target],
                eval_set=eval_set,
                verbose=100 if verbose else False,
            )

            if X_val is not None and y_val is not None:
                best_score = model.best_score
                print(
                    f"Best validation score (RMSE log-scale) for {target}: {best_score}"
                )

    def predict(self, X):
        """
        Generates predictions for both targets and applies inverse transformation.
        Returns a DataFrame with columns matching TARGET_COLS.
        """
        preds = {}
        for target in TARGET_COLS:
            model = self.models[target]
            # Predict in log space
            y_pred_log = model.predict(X)
            preds[target] = y_pred_log

        # Convert to DataFrame
        df_pred_log = pd.DataFrame(preds, index=X.index)

        # Apply inverse transform (exp(x) - 1) to get back to eV
        df_pred_original = inverse_transform_targets(df_pred_log)

        return df_pred_original

    def save_model(self, base_path):
        """
        Saves the internal XGBoost models to JSON files.
        """
        for target in TARGET_COLS:
            path = f"{base_path}_{target}.json"
            self.models[target].save_model(path)
            print(f"Model for {target} saved to {path}")

    def load_model(self, base_path):
        """
        Loads the internal XGBoost models from JSON files.
        """
        for target in TARGET_COLS:
            path = f"{base_path}_{target}.json"
            if os.path.exists(path):
                self.models[target].load_model(path)
                print(f"Model for {target} loaded from {path}")
            else:
                print(f"Warning: Model file {path} not found.")


def train_and_evaluate(load_cached_data=True):
    """
    Loads data, trains the dual model, and prints evaluation metrics.
    """
    print("Loading training and validation data...")
    X_train, y_train = load_data("train", load_cached_data=load_cached_data)
    X_val, y_val = load_data("val", load_cached_data=load_cached_data)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Initialize model
    model = DualTargetRegressor(params=XGB_PARAMS)

    # Train
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=100, verbose=True)

    # Evaluate on Validation Set
    print("\n--- Validation Evaluation ---")
    val_preds = model.predict(X_val)

    # Note: y_val is in log scale, val_preds is in original scale.
    # We need to inverse transform y_val for fair comparison in original units (eV)
    y_val_orig = inverse_transform_targets(y_val)

    for target in TARGET_COLS:
        mse = mean_squared_error(y_val_orig[target], val_preds[target])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_orig[target], val_preds[target])

        # Calculate RMSLE (Root Mean Squared Logarithmic Error) which is the competition metric
        # Since our model predicts in log space (log1p), the RMSE of the raw model output
        # is effectively the RMSLE of the original values.
        # Let's verify this using the raw log predictions from the model manually if needed,
        # but here we can just compute RMSLE on the original scale predictions.
        # RMSLE = sqrt(mean((log(1+p) - log(1+a))^2))
        rmsle = np.sqrt(
            mean_squared_error(
                np.log1p(val_preds[target]), np.log1p(y_val_orig[target])
            )
        )

        print(f"Target: {target}")
        print(f"  RMSE (eV): {rmse}")
        print(f"  MAE (eV):  {mae}")
        print(f"  RMSLE:     {rmsle}")

    # Save model
    model_save_path = os.path.join(WORKING_DIR, "xgb_model")
    model.save_model(model_save_path)

    return model


def generate_submission(model, load_cached_data=True):
    """
    Loads test data, generates predictions, and saves the submission file.
    """
    print("\n--- Generating Submission ---")
    X_test, _ = load_data("test", load_cached_data=load_cached_data)
    print(f"Test data shape: {X_test.shape}")

    # Predict
    preds_df = model.predict(X_test)

    # Load sample submission to ensure correct format and IDs
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # The test metadata should have the IDs. Let's load it to map index to ID if needed.
    # However, load_data preserves the index from metadata.
    # Let's verify alignment.
    test_meta = pd.read_csv(TEST_METADATA_PATH)

    # Ensure the predictions are aligned with the sample submission IDs
    # We map predictions to the 'id' column from test_meta
    preds_df["id"] = test_meta["id"].values

    # Create final submission dataframe
    submission = pd.DataFrame()
    submission["id"] = sample_sub["id"]

    # Merge predictions into submission based on ID
    submission = submission.merge(preds_df, on="id", how="left")

    # Fill any missing values (if any) with mean or 0 (should not happen)
    if submission.isnull().any().any():
        print("Warning: NaN values in submission. Filling with 0.")
        submission = submission.fillna(0)

    # Ensure column order matches sample submission
    submission = submission[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

    # Save
    print(f"Saving submission to {SUBMISSION_PATH}")
    submission.to_csv(SUBMISSION_PATH, index=False)

    # Preview
    print("Submission Head:")
    print(submission.head())


def run_pipeline(load_cached_data=True):
    """
    Executes the full training and submission pipeline.
    """
    model = train_and_evaluate(load_cached_data=load_cached_data)
    generate_submission(model, load_cached_data=load_cached_data)
