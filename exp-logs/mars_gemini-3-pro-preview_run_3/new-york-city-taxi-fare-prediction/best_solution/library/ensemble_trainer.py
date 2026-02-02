import os
import gc
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SEED,
)
from library.data_loader import load_data, get_stacking_splits
from library.feature_engineering import FeatureEngineer
from library.model_definitions import (
    get_xgboost_learner,
    get_lightgbm_learner,
    get_meta_learner,
)

# Ensure reproducibility
np.random.seed(SEED)


class EnsembleTrainer:
    """
    Orchestrates the training of a two-stage stacked ensemble for taxi fare prediction.
    Stage 0: XGBoost (GPU) and LightGBM (CPU).
    Stage 1: Ridge Regression (Meta-learner).
    """

    def __init__(self):
        self.xgb_model = None
        self.lgbm_model = None
        self.meta_model = None
        self.feature_engineer = FeatureEngineer()

    def train_stack(self, load_cached_data=True, debug=False, debug_size=100000):
        """
        Executes the full stacking pipeline: Data Loading -> Splitting -> Feature Engineering
        -> Base Model Training -> Meta Model Training -> Evaluation -> Submission.

        Args:
            load_cached_data (bool): If True, attempts to load preprocessed data from cache.
            debug (bool): If True, subsamples the dataset for rapid debugging.
            debug_size (int): Number of rows to use when debug is True.

        Returns:
            dict: Dictionary containing trained models and validation metrics.
        """
        # 1. Load Data
        # load_data handles caching of raw cleaned data
        train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

        # Handle Debugging Subsampling
        suffix = ""
        if debug:
            print(f"DEBUG MODE: Sampling {debug_size} rows...")
            train_df = train_df.sample(
                min(len(train_df), debug_size), random_state=SEED
            )
            val_df = val_df.sample(min(len(val_df), debug_size), random_state=SEED)
            suffix = "_debug"

        # 2. Split Training Data for Stacking (Base vs Meta)
        # We split the main training set. val_df is kept separate for final evaluation/early stopping.
        print("Splitting training data into Base and Meta sets...")
        base_train_df, meta_train_df = get_stacking_splits(train_df, val_size=0.1)

        # Free memory from original train_df
        del train_df
        gc.collect()

        # 3. Feature Engineering
        # We process datasets sequentially and rely on the FeatureEngineer's internal caching.
        # We pass distinct names to avoid cache collisions between debug and full runs.
        print("Engineering features for Base Train...")
        base_train_df = self.feature_engineer.process(
            base_train_df, f"base_train{suffix}"
        )

        print("Engineering features for Meta Train...")
        meta_train_df = self.feature_engineer.process(
            meta_train_df, f"meta_train{suffix}"
        )

        print("Engineering features for Validation...")
        val_df = self.feature_engineer.process(val_df, f"val{suffix}")

        print("Engineering features for Test...")
        test_df = self.feature_engineer.process(test_df, f"test{suffix}")

        # 4. Prepare Feature Matrices (X and y)
        target_col = "fare_amount"
        # Columns to exclude from features
        ignore_cols = ["key", "fare_amount", "pickup_datetime"]

        def get_features_target(df):
            feats = [c for c in df.columns if c not in ignore_cols]
            return df[feats], df[target_col]

        X_base, y_base = get_features_target(base_train_df)
        X_meta, y_meta = get_features_target(meta_train_df)
        X_val, y_val = get_features_target(val_df)

        print(f"Base Train shape: {X_base.shape}")
        print(f"Meta Train shape: {X_meta.shape}")

        # 5. Train Base Learners (Level 0)

        # Optimization: Use a subset of validation data for early stopping to speed up evaluation
        # while preserving the full validation set for final scoring.
        if len(X_val) > 500000:
            print(
                f"Subsampling validation set for early stopping (from {len(X_val)} to 500000)..."
            )
            # Use numpy random choice for indices to keep X and y aligned
            idx = np.random.choice(len(X_val), 500000, replace=False)
            X_val_eval = X_val.iloc[idx]
            y_val_eval = y_val.iloc[idx]
        else:
            X_val_eval = X_val
            y_val_eval = y_val

        # --- XGBoost ---
        print("\nTraining XGBoost Base Learner...")
        self.xgb_model = get_xgboost_learner(
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        # XGBoost supports early_stopping_rounds in fit()
        self.xgb_model.fit(
            X_base,
            y_base,
            eval_set=[(X_val_eval, y_val_eval)],
            verbose=VERBOSE_EVAL,
        )

        # --- LightGBM ---
        print("\nTraining LightGBM Base Learner...")
        self.lgbm_model = get_lightgbm_learner()
        # LightGBM uses callbacks for early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]
        self.lgbm_model.fit(
            X_base,
            y_base,
            eval_set=[(X_val_eval, y_val_eval)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # 6. Generate Meta Features (Level 1 Input)
        print("\nGenerating Out-of-Sample predictions for Meta Learner...")
        # We predict on the Meta Train set using the trained base models
        xgb_pred_meta = self.xgb_model.predict(X_meta)
        lgbm_pred_meta = self.lgbm_model.predict(X_meta)

        # Stack predictions to form the input matrix for the meta-learner
        X_stack_meta = np.column_stack((xgb_pred_meta, lgbm_pred_meta))

        # 7. Train Meta Learner (Level 1)
        print("Training Ridge Meta Learner...")
        self.meta_model = get_meta_learner()
        self.meta_model.fit(X_stack_meta, y_meta)

        print("Meta Learner Coefficients:", self.meta_model.coef_)
        print("Meta Learner Intercept:", self.meta_model.intercept_)

        # 8. Final Evaluation on Validation Set
        print("\nEvaluating Ensemble on Validation Set...")
        # Generate base predictions for validation set
        xgb_pred_val = self.xgb_model.predict(X_val)
        lgbm_pred_val = self.lgbm_model.predict(X_val)
        X_stack_val = np.column_stack((xgb_pred_val, lgbm_pred_val))

        # Generate final ensemble prediction
        final_pred_val = self.meta_model.predict(X_stack_val)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, final_pred_val))
        print(f"Final Ensemble Validation RMSE: {rmse}")

        # Save Models
        self._save_models(suffix)

        # 9. Generate Submission
        self._generate_submission(test_df)

        return {
            "xgb": self.xgb_model,
            "lgbm": self.lgbm_model,
            "meta": self.meta_model,
            "rmse": rmse,
        }

    def _save_models(self, suffix=""):
        """Saves trained models to the working directory."""
        print("Saving models...")
        joblib.dump(
            self.xgb_model, os.path.join(WORKING_DIR, f"xgb_model{suffix}.joblib")
        )
        joblib.dump(
            self.lgbm_model, os.path.join(WORKING_DIR, f"lgbm_model{suffix}.joblib")
        )
        joblib.dump(
            self.meta_model, os.path.join(WORKING_DIR, f"meta_model{suffix}.joblib")
        )

    def _generate_submission(self, test_df):
        """Generates and saves the submission CSV file."""
        print("Generating submission file...")
        ignore_cols = ["key", "fare_amount", "pickup_datetime"]
        features = [c for c in test_df.columns if c not in ignore_cols]
        X_test = test_df[features]

        # Base Predictions
        xgb_pred_test = self.xgb_model.predict(X_test)
        lgbm_pred_test = self.lgbm_model.predict(X_test)

        # Stack
        X_stack_test = np.column_stack((xgb_pred_test, lgbm_pred_test))

        # Meta Prediction
        final_pred_test = self.meta_model.predict(X_stack_test)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"key": test_df["key"], "fare_amount": final_pred_test}
        )

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
