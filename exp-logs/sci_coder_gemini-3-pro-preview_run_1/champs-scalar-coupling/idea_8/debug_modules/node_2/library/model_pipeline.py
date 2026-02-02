import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gc
from library.config import MODEL_DIR, get_xgb_params, COUPLING_TYPES, RANDOM_STATE
from library.utils import save_submission, calculate_log_mae
from library.feature_engineering import get_data


class StratifiedRegressor:
    """
    A wrapper class that manages an ensemble of XGBoost models,
    one for each scalar coupling type.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model_path(self, coupling_type):
        return os.path.join(self.model_dir, f"{coupling_type}.json")

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Trains a separate XGBoost model for each coupling type found in the data.
        """
        print(f"Starting Stratified Training on {len(X_train)} samples...")

        # Ensure y is accessible by index
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Iterate over each known coupling type
        for c_type in COUPLING_TYPES:
            print(f"\n--- Processing Coupling Type: {c_type} ---")

            # Create boolean masks
            mask_train = X_train["type"] == c_type
            mask_val = X_val["type"] == c_type

            # Skip if no data for this type
            if not mask_train.any():
                print(f"No training data for {c_type}. Skipping.")
                continue

            # Slice data and drop the 'type' column (it's constant and non-numeric)
            # We use .values for y to ensure alignment
            X_t = X_train.loc[mask_train].drop(columns=["type"])
            y_t = y_train[mask_train.values]

            X_v = X_val.loc[mask_val].drop(columns=["type"])
            y_v = y_val[mask_val.values]

            print(f"Train shape: {X_t.shape}, Val shape: {X_v.shape}")

            # Configure XGBoost
            params = get_xgb_params(c_type)
            train_params = params.pop("training", {})  # Extract training-specific args

            dtrain = xgb.DMatrix(X_t, label=y_t)
            dval = xgb.DMatrix(X_v, label=y_v)

            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=train_params.get("n_estimators", 10000),
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=train_params.get("early_stopping_rounds", 50),
                verbose_eval=train_params.get("verbose", 100),
            )

            # Save model
            model_path = self._get_model_path(c_type)
            model.save_model(model_path)
            self.models[c_type] = model
            print(f"Model for {c_type} saved to {model_path}")

            # Clean up memory
            del X_t, y_t, X_v, y_v, dtrain, dval, model
            gc.collect()

    def predict(self, X_test):
        """
        Generates predictions by routing samples to their specific type-based model.
        """
        print(f"Starting Stratified Prediction on {len(X_test)} samples...")

        # Initialize empty prediction array aligned with input index
        predictions = pd.Series(index=X_test.index, dtype=float)
        predictions[:] = np.nan

        for c_type in COUPLING_TYPES:
            mask_test = X_test["type"] == c_type

            if not mask_test.any():
                continue

            # Slice
            X_t = X_test.loc[mask_test].drop(columns=["type"])

            # Load model
            model_path = self._get_model_path(c_type)
            if c_type in self.models:
                model = self.models[c_type]
            elif os.path.exists(model_path):
                model = xgb.Booster()
                model.load_model(model_path)
                self.models[c_type] = model
            else:
                raise FileNotFoundError(
                    f"No trained model found for {c_type} at {model_path}"
                )

            # Predict
            dtest = xgb.DMatrix(X_t)
            preds = model.predict(dtest)

            # Assign back to main array
            predictions.loc[mask_test] = preds

            # Clean up
            del X_t, dtest
            gc.collect()

        return predictions.values


def train_and_predict(load_cached_data=True):
    """
    Main pipeline function:
    1. Loads data (using caching from FeatureEngineer).
    2. Trains stratified models.
    3. Evaluates on Validation set.
    4. Generates Submission for Test set.
    """
    # 1. Load Data
    print("Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, ids_test = get_data(
        load_cached_data=load_cached_data
    )

    # 2. Train
    regressor = StratifiedRegressor()
    regressor.fit(X_train, y_train, X_val, y_val)

    # 3. Validation Evaluation
    print("\nRunning Validation Evaluation...")
    val_preds = regressor.predict(X_val)

    # Construct evaluation dataframe
    val_eval_df = pd.DataFrame(
        {
            "type": X_val["type"].values,
            "scalar_coupling_constant": y_val,
            "prediction": val_preds,
        }
    )

    # Calculate and print metric
    calculate_log_mae(val_eval_df)

    # 4. Test Prediction & Submission
    print("\nGenerating Test Predictions...")
    test_preds = regressor.predict(X_test)

    save_submission(ids_test, test_preds)

    print("Pipeline completed successfully.")
