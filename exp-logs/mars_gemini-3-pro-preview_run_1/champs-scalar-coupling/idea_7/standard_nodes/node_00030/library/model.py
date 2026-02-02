import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import (
    COUPLING_TYPES,
    XGB_PARAMS,
    MODEL_SAVE_DIR,
    SUBMISSION_FILE_PATH,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
)
from library.utils import calculate_log_mae, print_full_precision_metrics


class StratifiedModel:
    """
    Implements a Stratified Gradient Boosting approach where a separate
    XGBoost model is trained for each scalar coupling type.
    """

    def __init__(self):
        """
        Initialize the StratifiedModel.
        Ensures the model save directory exists.
        """
        self.model_dir = MODEL_SAVE_DIR
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains stratified XGBoost models for each coupling type.

        Args:
            X_train (pd.DataFrame): Training features (must include 'type').
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features (must include 'type').
            y_val (pd.Series): Validation targets.
        """
        print("Starting Stratified Training...")

        overall_metrics = []

        for c_type in COUPLING_TYPES:
            print(f"\n{'='*30}\nTraining Model for Type: {c_type}\n{'='*30}")

            # 1. Stratify Data
            train_mask = X_train["type"] == c_type
            val_mask = X_val["type"] == c_type

            if not train_mask.any():
                print(f"Warning: No training data found for {c_type}. Skipping.")
                continue

            # Filter and drop 'type' column as it is not a feature for the model
            X_t = X_train.loc[train_mask].drop(columns=["type"])
            y_t = y_train.loc[train_mask]

            X_v = X_val.loc[val_mask].drop(columns=["type"])
            y_v = y_val.loc[val_mask]

            print(f"Train Shape: {X_t.shape}, Val Shape: {X_v.shape}")

            # 2. Initialize Model
            # We use the hyperparameters from config.py optimized for A100
            model = xgb.XGBRegressor(**XGB_PARAMS)

            # 3. Train with Early Stopping
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=VERBOSE_EVAL)

            # 4. Evaluate
            y_pred_val = model.predict(X_v)
            score, breakdown = calculate_log_mae(
                y_v, y_pred_val, pd.Series([c_type] * len(y_v)), verbose=False
            )

            # Print Full Precision Metric
            print_full_precision_metrics(
                {"Log MAE": score}, prefix=f"[{c_type}] Validation "
            )
            overall_metrics.append(score)

            # 5. Save Model
            model_path = os.path.join(self.model_dir, f"{c_type}.json")
            model.save_model(model_path)
            print(f"Model saved to {model_path}")

            # 6. Cleanup to free memory
            del X_t, y_t, X_v, y_v, model, y_pred_val
            gc.collect()

        if overall_metrics:
            avg_score = np.mean(overall_metrics)
            print("\n" + "=" * 30)
            print_full_precision_metrics(
                {"Average Log MAE": avg_score}, prefix="[Overall] "
            )
            print("=" * 30)

    def predict(self, X_test, test_ids):
        """
        Generates predictions for the test set using the stratified models.

        Args:
            X_test (pd.DataFrame): Test features (must include 'type').
            test_ids (pd.Series): IDs corresponding to the test rows.

        Returns:
            pd.DataFrame: Submission dataframe with 'id' and 'scalar_coupling_constant'.
        """
        print("\nStarting Stratified Inference...")
        submission_parts = []

        for c_type in COUPLING_TYPES:
            # 1. Stratify Data
            mask = X_test["type"] == c_type

            if not mask.any():
                continue

            X_t = X_test.loc[mask].drop(columns=["type"])
            ids_t = test_ids.loc[mask]

            # 2. Load Model
            model_path = os.path.join(self.model_dir, f"{c_type}.json")
            if not os.path.exists(model_path):
                print(f"Error: Model for {c_type} not found at {model_path}. Skipping.")
                continue

            model = xgb.XGBRegressor()
            model.load_model(model_path)

            # Ensure the model uses the GPU for inference if available
            if XGB_PARAMS.get("device") == "cuda":
                model.set_params(device="cuda")

            # 3. Predict
            print(f"Predicting for {c_type} (n={len(X_t)})...")
            y_pred = model.predict(X_t)

            # 4. Collect Results
            sub_part = pd.DataFrame({"id": ids_t, "scalar_coupling_constant": y_pred})
            submission_parts.append(sub_part)

            # Cleanup
            del X_t, model, y_pred
            gc.collect()

        # 5. Aggregate and Save
        if submission_parts:
            submission = pd.concat(submission_parts).sort_values("id")

            # Ensure directory exists
            os.makedirs(os.path.dirname(SUBMISSION_FILE_PATH), exist_ok=True)

            submission.to_csv(SUBMISSION_FILE_PATH, index=False)
            print(f"Submission saved to {SUBMISSION_FILE_PATH}")
            print(f"Submission Shape: {submission.shape}")
            return submission
        else:
            print("Warning: No predictions generated.")
            return pd.DataFrame(columns=["id", "scalar_coupling_constant"])
