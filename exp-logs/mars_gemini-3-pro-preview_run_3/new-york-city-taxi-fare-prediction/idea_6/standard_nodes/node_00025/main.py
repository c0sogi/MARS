import os
import sys
import gc
import random
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Import library modules
from library.config import Config
from library.data_processing import process_data
from library.model_factory import ModelFactory
from library.training import generate_submission


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting runfile.py execution...")

    # 2. Data Loading
    # We need the full validation set for the metric calculation as per requirements.
    # Therefore, we pass debug_sample_size=None to process_data to ensure val_df is complete.
    # We will sample train_df manually in the next step for the fast baseline.
    print("Loading and processing data...")
    train_df, val_df, test_df = process_data(
        load_cached_data=True, debug_sample_size=None
    )

    # 3. Prepare Training Data (Full Scale)
    # Cite solution_lesson_node_00012: Data Volume Dominance.
    print(f"Using full training set: {len(train_df)} rows.")

    # Define features (exclude ID, target, and raw datetime)
    exclude_cols = ["key", "fare_amount", "pickup_datetime"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features used: {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df["fare_amount"]

    # Validation data (Full)
    X_val = val_df[feature_cols]
    y_val = val_df["fare_amount"]

    # Free up memory from full train_df
    del train_df
    gc.collect()

    # 4. Model Training
    models = {}

    # -- Train XGBoost --
    # Cite solution_lesson_node_00010: Scaling Tree Capacity.
    if Config.ENSEMBLE_WEIGHTS.get("xgb", 0) > 0:
        print("\nTraining XGBoost (Full Capacity)...")
        # Create model with default Config params
        xgb_model = ModelFactory.create_xgboost()

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=Config.VERBOSE_EVAL,
        )
        models["xgb"] = xgb_model

        # Save model
        xgb_path = os.path.join(Config.WORKING_DIR, "xgboost_model.joblib")
        joblib.dump(xgb_model, xgb_path)
        print(f"XGBoost model saved to {xgb_path}")

    # -- Train LightGBM --
    if Config.ENSEMBLE_WEIGHTS.get("lgbm", 0) > 0:
        print("\nTraining LightGBM (Full Capacity)...")
        # Create model with default Config params
        lgbm_model = ModelFactory.create_lgbm()

        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        lgbm_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        models["lgbm"] = lgbm_model

        # Save model
        lgbm_path = os.path.join(Config.WORKING_DIR, "lgbm_model.joblib")
        joblib.dump(lgbm_model, lgbm_path)
        print(f"LightGBM model saved to {lgbm_path}")

    # 5. Evaluation
    print("\nEvaluating Ensemble on Full Validation Set...")
    val_preds = np.zeros(len(X_val))
    total_weight = 0.0

    for name, model in models.items():
        weight = Config.ENSEMBLE_WEIGHTS.get(name, 0.0)
        if weight > 0:
            print(f"Predicting with {name}...")
            pred = model.predict(X_val)
            val_preds += pred * weight
            total_weight += weight

    if total_weight > 0:
        val_preds /= total_weight

    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    # Required output format
    print(f"Final Validation Metric: {final_rmse}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    residuals = np.abs(y_val - val_preds)

    # Calculate correlation between features and error magnitude
    # We select numeric columns for correlation
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns
    analysis_df = X_val[numeric_cols].copy()
    analysis_df["error_magnitude"] = residuals

    correlations = (
        analysis_df.corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top Feature Correlations with Error Magnitude:")
    # Exclude self-correlation
    print(correlations.drop("error_magnitude", errors="ignore").head(5))

    del analysis_df
    gc.collect()

    # 7. Submission
    THRESHOLD = 3.3935366001817666
    if final_rmse < THRESHOLD:
        print(
            f"\nMetric {final_rmse} < Threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(models, test_df, feature_cols)
    else:
        print(f"\nMetric {final_rmse} >= Threshold {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
