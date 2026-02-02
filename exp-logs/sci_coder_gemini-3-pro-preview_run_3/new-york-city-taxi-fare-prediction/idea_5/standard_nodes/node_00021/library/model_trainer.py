import os
import gc
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from library.config import (
    WORKING_DIR,
    SUBMISSION_FILE,
    XGB_PARAMS,
    LGBM_PARAMS,
    WEIGHT_XGB,
    WEIGHT_LGBM,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    RANDOM_SEED,
)


class ModelTrainer:
    """
    Manages the training, evaluation, and inference of the ensemble models.
    """

    def __init__(self):
        self.xgb_model_path = os.path.join(WORKING_DIR, "xgboost_model.joblib")
        self.lgbm_model_path = os.path.join(WORKING_DIR, "lgbm_model.joblib")
        self.xgb_model = None
        self.lgbm_model = None

    def train_xgboost(self, X_train, y_train, X_val, y_val, load_cached_model=True):
        """
        Trains the XGBoost model using GPU acceleration.
        """
        # 1. Check Cache
        if load_cached_model and os.path.exists(self.xgb_model_path):
            print(f"Loading cached XGBoost model from {self.xgb_model_path}...")
            self.xgb_model = joblib.load(self.xgb_model_path)
            return

        print("Training XGBoost model (GPU)...")

        # Train with Early Stopping
        # Using callbacks for compatibility with newer versions
        early_stop = xgb.callback.EarlyStopping(
            rounds=EARLY_STOPPING_ROUNDS,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name="rmse",
        )

        # Log evaluation
        log_eval = xgb.callback.EvaluationMonitor(rank=0, period=VERBOSE_EVAL)

        # Initialize Regressor
        # Note: XGBoost 2.x/3.x sklearn API
        model = xgb.XGBRegressor(callbacks=[early_stop, log_eval], **XGB_PARAMS)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,  # Handled by callback
        )

        self.xgb_model = model

        # Save Model
        print(f"Saving XGBoost model to {self.xgb_model_path}...")
        joblib.dump(self.xgb_model, self.xgb_model_path)

        # Validation Score
        preds = self.xgb_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"XGBoost Validation RMSE: {rmse}")

        gc.collect()

    def train_lgbm(self, X_train, y_train, X_val, y_val, load_cached_model=True):
        """
        Trains the LightGBM model using CPU.
        """
        # 1. Check Cache
        if load_cached_model and os.path.exists(self.lgbm_model_path):
            print(f"Loading cached LightGBM model from {self.lgbm_model_path}...")
            self.lgbm_model = joblib.load(self.lgbm_model_path)
            return

        print("Training LightGBM model (CPU)...")

        # Initialize Regressor
        model = lgb.LGBMRegressor(**LGBM_PARAMS)

        # Callbacks for LightGBM 4.x
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        # Train
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        self.lgbm_model = model

        # Save Model
        print(f"Saving LightGBM model to {self.lgbm_model_path}...")
        joblib.dump(self.lgbm_model, self.lgbm_model_path)

        # Validation Score
        preds = self.lgbm_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"LightGBM Validation RMSE: {rmse}")

        gc.collect()

    def predict_ensemble(self, X_test):
        """
        Generates predictions using the weighted average of XGBoost and LightGBM.
        """
        if self.xgb_model is None:
            if os.path.exists(self.xgb_model_path):
                self.xgb_model = joblib.load(self.xgb_model_path)
            else:
                raise ValueError("XGBoost model not trained or found.")

        if self.lgbm_model is None:
            if os.path.exists(self.lgbm_model_path):
                self.lgbm_model = joblib.load(self.lgbm_model_path)
            else:
                raise ValueError("LightGBM model not trained or found.")

        print("Generating ensemble predictions...")

        # XGBoost Predictions
        xgb_preds = self.xgb_model.predict(X_test)

        # LightGBM Predictions
        lgbm_preds = self.lgbm_model.predict(X_test)

        # Weighted Average
        final_preds = (WEIGHT_XGB * xgb_preds) + (WEIGHT_LGBM * lgbm_preds)

        return final_preds

    def save_submission(self, keys, predictions):
        """
        Saves the predictions to a CSV file in the required format.
        """
        print(f"Saving submission to {SUBMISSION_FILE}...")

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

        # Create DataFrame
        submission_df = pd.DataFrame({"key": keys, "fare_amount": predictions})

        # Save
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print("Submission saved successfully.")
