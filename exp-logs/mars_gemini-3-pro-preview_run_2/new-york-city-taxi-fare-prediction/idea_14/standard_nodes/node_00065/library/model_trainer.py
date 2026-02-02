import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    XGB_PARAMS,
    TRAIN_SUBSAMPLE_SIZE,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
)
from library.data_processor import get_processed_data
from library.feature_engineering import InteractionStatsEngine


class XGBoostTrainer:
    """
    Wrapper class for XGBoost Regressor to handle training, saving, loading, and prediction.
    """

    def __init__(self, params=None, model_path=None):
        self.params = params if params else XGB_PARAMS
        self.model_path = (
            model_path if model_path else os.path.join(WORKING_DIR, "xgb_model.json")
        )
        self.model = None

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    ):
        """
        Trains the XGBoost model with early stopping.
        """
        print(f"Initializing training with {X_train.shape[0]} samples...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals_result = {}

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
        )

        # Print full precision of the best score
        best_score = self.model.best_score
        print(f"Best Validation RMSE (Full Precision): {best_score}")

        self.save_model()

        # Cleanup
        del dtrain, dval
        gc.collect()

    def predict(self, X_test):
        """
        Generates predictions using the trained model.
        """
        if self.model is None:
            self.load_model()

        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
        return predictions

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        if self.model:
            print(f"Saving model to {self.model_path}...")
            self.model.save_model(self.model_path)

    def load_model(self):
        """
        Loads the model from disk.
        """
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")


def run_pipeline(
    subsample_size=TRAIN_SUBSAMPLE_SIZE,
    num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    load_cached_data=True,
):
    """
    Orchestrates the full training and prediction pipeline.

    1. Wisdom Phase: Generate Global Stats from Strict Data.
    2. Learner Phase: Prepare Training Data with Interaction Features.
    3. Validation Prep: Prepare Validation Data.
    4. Training: Train XGBoost Model.
    5. Prediction: Predict on Test Data and Save Submission.
    """

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Step 1: Wisdom Phase (Global Stats Generation) ---
    print("\n=== Phase 1: Wisdom (Global Stats Generation) ===")
    # Load strict data (cached if available)
    df_strict = get_processed_data(
        "train", mode="strict", load_cached_data=load_cached_data
    )

    # Initialize and Fit Engine
    engine = InteractionStatsEngine(working_dir=WORKING_DIR)
    engine.fit(df_strict, load_cached=load_cached_data)

    # Free memory immediately
    del df_strict
    gc.collect()

    # --- Step 2: Learner Phase (Training Data Prep) ---
    print("\n=== Phase 2: Learner (Training Data Prep) ===")
    df_train = get_processed_data(
        "train",
        mode="loose",
        subsample_size=subsample_size,
        load_cached_data=load_cached_data,
    )
    df_train = engine.transform_train(df_train)

    # Define features
    # Exclude metadata, target, fold info, and raw geohash strings
    exclude_cols = {"key", "fare_amount", "pickup_datetime", "fold_id"}
    exclude_cols.update([c for c in df_train.columns if "geohash" in c])

    features = [c for c in df_train.columns if c not in exclude_cols]
    target = "fare_amount"

    print(f"Selected Features ({len(features)}): {features}")

    # --- Step 3: Validation Data Prep ---
    print("\n=== Phase 3: Validation Data Prep ===")
    # Use loose filtering for validation to match training distribution
    df_val = get_processed_data("val", mode="loose", load_cached_data=load_cached_data)
    df_val = engine.transform_test(df_val)

    # --- Step 4: Model Training ---
    print("\n=== Phase 4: Model Training ===")
    X_train = df_train[features]
    y_train = df_train[target]
    X_val = df_val[features]
    y_val = df_val[target]

    trainer = XGBoostTrainer()
    trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Free memory
    del df_train, df_val, X_train, y_train, X_val, y_val
    gc.collect()

    # --- Step 5: Prediction ---
    print("\n=== Phase 5: Prediction ===")
    # Test data uses 'inference' mode (no filtering)
    df_test = get_processed_data(
        "test", mode="inference", load_cached_data=load_cached_data
    )
    df_test = engine.transform_test(df_test)

    X_test = df_test[features]

    print("Generating predictions...")
    preds = trainer.predict(X_test)

    # Post-processing: Floor at $2.50
    preds = np.maximum(preds, 2.50)

    # Save Submission
    submission = pd.DataFrame({"key": df_test["key"], "fare_amount": preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Pipeline Complete.")
